import logging
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional

from ..train import modelconfigs

# Cache for symmetry permutations used by pairwise targets.
_SYMM_FLAT_PERM_CACHE = {}


def _get_symmetry_flat_perm(pos_len: int, symm: int, device: torch.device) -> torch.Tensor:
    """Return a LongTensor perm of shape [pos_len*pos_len] mapping new_flat_idx -> old_flat_idx."""
    key = (pos_len, symm, device.type, device.index)
    perm = _SYMM_FLAT_PERM_CACHE.get(key)
    if perm is not None:
        return perm
    base = torch.arange(pos_len * pos_len, device=device, dtype=torch.int64).view(pos_len, pos_len)
    perm = apply_symmetry(base, symm).reshape(-1).contiguous()
    _SYMM_FLAT_PERM_CACHE[key] = perm
    return perm


def apply_symmetry_connection_targets(tensor: torch.Tensor, symm: int, pos_len: int) -> torch.Tensor:
    """Apply symmetry to connectionTargetsNPP (N, Pos, Pos) by permuting both axes."""
    assert tensor.ndim == 3
    assert tensor.shape[1] == pos_len * pos_len
    assert tensor.shape[2] == pos_len * pos_len
    perm = _get_symmetry_flat_perm(pos_len, symm, tensor.device)
    return tensor[:, perm, :][:, :, perm]

def read_npz_training_data(
    npz_files,
    batch_size: int,
    world_size: int,
    rank: int,
    pos_len: int,
    device,
    randomize_symmetries: bool,
    include_meta: bool,
    model_config: modelconfigs.ModelConfig,
):
    rand = np.random.default_rng(seed=list(os.urandom(12)))
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    (h_base,h_builder) = build_history_matrices(model_config, device)

    include_qvalues = model_config["version"] >= 16
    include_weighted_values = model_config["version"] >= 18
    include_connection_targets = bool(model_config.get("connection_head", {}).get("enabled", False))

    def load_npz_file(npz_file):
        with np.load(npz_file) as npz:
            binaryInputNCHWPacked = npz["binaryInputNCHWPacked"]
            globalInputNC = npz["globalInputNC"]
            policyTargetsNCMove = npz["policyTargetsNCMove"].astype(np.float32)
            globalTargetsNC = npz["globalTargetsNC"]
            scoreDistrN = npz["scoreDistrN"].astype(np.float32)
            valueTargetsNCHW = npz["valueTargetsNCHW"].astype(np.float32)
            if include_meta:
                metadataInputNC = npz["metadataInputNC"].astype(np.float32)
            else:
                metadataInputNC = None
            if include_qvalues and "qValueTargetsNCMove" in npz:
                qValueTargetsNCMove = npz["qValueTargetsNCMove"].astype(np.float32)
            else:
                qValueTargetsNCMove = None
            if include_weighted_values and "weightedValueTargetsNC" in npz:
                weightedValueTargetsNC = npz["weightedValueTargetsNC"].astype(np.float32)
            else:
                weightedValueTargetsNC = None

            if include_connection_targets and "connectionTargetsNPP" in npz:
                connectionTargetsNPP = npz["connectionTargetsNPP"].astype(np.float32)
            else:
                connectionTargetsNPP = None
        del npz

        binaryInputNCHW = np.unpackbits(binaryInputNCHWPacked,axis=2)
        assert len(binaryInputNCHW.shape) == 3
        assert binaryInputNCHW.shape[2] == ((pos_len * pos_len + 7) // 8) * 8
        binaryInputNCHW = binaryInputNCHW[:,:,:pos_len*pos_len]
        binaryInputNCHW = np.reshape(binaryInputNCHW, (
            binaryInputNCHW.shape[0], binaryInputNCHW.shape[1], pos_len, pos_len
        )).astype(np.float32)

        # Handle mismatch between data (possibly 23 channels including weight mask) and model config (possibly 22)
        if binaryInputNCHW.shape[1] != num_bin_features:
            if binaryInputNCHW.shape[1] == 23 and num_bin_features == 22:
                # Drop the last channel (weight mask) so older nets (version<17) can still train
                binaryInputNCHW = binaryInputNCHW[:, :22]
                if not hasattr(read_npz_training_data, "_warned_feature_crop"):
                    logging.warning("Cropping weight mask feature channel (23->22) to match older model version; upgrade model config to version 17+ to use it.")
                    read_npz_training_data._warned_feature_crop = True
            elif binaryInputNCHW.shape[1] == 22 and num_bin_features == 23:
                # Pad with a weight mask of 1.0s (neutral) for older data when using newer model
                pad = np.ones((binaryInputNCHW.shape[0],1,pos_len,pos_len), dtype=binaryInputNCHW.dtype)
                binaryInputNCHW = np.concatenate([binaryInputNCHW, pad], axis=1)
                if not hasattr(read_npz_training_data, "_warned_feature_pad"):
                    logging.warning("Padding missing weight mask feature channel (22->23) with ones; regenerate data with version 17 features for full effect.")
                    read_npz_training_data._warned_feature_pad = True
            else:
                raise AssertionError(f"Binary feature channel mismatch: data has {binaryInputNCHW.shape[1]}, model expects {num_bin_features}")
        assert binaryInputNCHW.shape[1] == num_bin_features
        assert globalInputNC.shape[1] == num_global_features
        return (
            npz_file,
            binaryInputNCHW,
            globalInputNC,
            policyTargetsNCMove,
            globalTargetsNC,
            scoreDistrN,
            valueTargetsNCHW,
            metadataInputNC,
            qValueTargetsNCMove,
            weightedValueTargetsNC,
            connectionTargetsNPP,
        )

    if not npz_files:
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_npz_file, npz_files[0])

        for next_file in (npz_files[1:] + [None]):
            (
                npz_file,
                binaryInputNCHW,
                globalInputNC,
                policyTargetsNCMove,
                globalTargetsNC,
                scoreDistrN,
                valueTargetsNCHW,
                metadataInputNC,
                qValueTargetsNCMove,
                weightedValueTargetsNC,
                connectionTargetsNPP,
            ) = future.result()

            num_samples = binaryInputNCHW.shape[0]
            # Just discard stuff that doesn't divide evenly
            num_whole_steps = num_samples // (batch_size * world_size)

            logging.info(f"Beginning {npz_file} with {num_whole_steps * world_size} usable batches, my rank is {rank}")

            if next_file is not None:
                logging.info(f"Preloading {next_file} while processing this file")
                future = executor.submit(load_npz_file, next_file)

            for n in range(num_whole_steps):
                start = (n * world_size + rank) * batch_size
                end = start + batch_size

                batch_binaryInputNCHW = torch.from_numpy(binaryInputNCHW[start:end]).to(device)
                batch_globalInputNC = torch.from_numpy(globalInputNC[start:end]).to(device)
                batch_policyTargetsNCMove = torch.from_numpy(policyTargetsNCMove[start:end]).to(device)
                batch_globalTargetsNC = torch.from_numpy(globalTargetsNC[start:end]).to(device)
                batch_scoreDistrN = torch.from_numpy(scoreDistrN[start:end]).to(device)
                batch_valueTargetsNCHW = torch.from_numpy(valueTargetsNCHW[start:end]).to(device)
                if include_meta:
                    batch_metadataInputNC = torch.from_numpy(metadataInputNC[start:end]).to(device)
                if include_qvalues and qValueTargetsNCMove is not None:
                    batch_qValueTargetsNCMove = torch.from_numpy(qValueTargetsNCMove[start:end]).to(device)
                if include_weighted_values and weightedValueTargetsNC is not None:
                    batch_weightedValueTargetsNC = torch.from_numpy(weightedValueTargetsNC[start:end]).to(device)
                if include_connection_targets and connectionTargetsNPP is not None:
                    batch_connectionTargetsNPP = torch.from_numpy(connectionTargetsNPP[start:end]).to(device)

                (batch_binaryInputNCHW, batch_globalInputNC) = apply_history_matrices(
                    model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder
                )

                if randomize_symmetries:
                    symm = int(rand.integers(0,8))
                    batch_binaryInputNCHW = apply_symmetry(batch_binaryInputNCHW, symm)
                    batch_policyTargetsNCMove = apply_symmetry_policy(batch_policyTargetsNCMove, symm, pos_len)
                    batch_valueTargetsNCHW = apply_symmetry(batch_valueTargetsNCHW, symm)
                    if include_qvalues and qValueTargetsNCMove is not None:
                        batch_qValueTargetsNCMove = apply_symmetry_policy(batch_qValueTargetsNCMove, symm, pos_len)
                    if include_weighted_values and weightedValueTargetsNC is not None:
                        # Weighted targets are [N,5] no symmetry needed
                        pass
                    if include_connection_targets and connectionTargetsNPP is not None:
                        batch_connectionTargetsNPP = apply_symmetry_connection_targets(batch_connectionTargetsNPP, symm, pos_len)

                batch_binaryInputNCHW = batch_binaryInputNCHW.contiguous()
                batch_policyTargetsNCMove = batch_policyTargetsNCMove.contiguous()
                batch_valueTargetsNCHW = batch_valueTargetsNCHW.contiguous()
                if include_qvalues and qValueTargetsNCMove is not None:
                    batch_qValueTargetsNCMove = batch_qValueTargetsNCMove.contiguous()
                if include_weighted_values and weightedValueTargetsNC is not None:
                    batch_weightedValueTargetsNC = batch_weightedValueTargetsNC.contiguous()
                if include_connection_targets and connectionTargetsNPP is not None:
                    batch_connectionTargetsNPP = batch_connectionTargetsNPP.contiguous()

                batch = dict(
                    binaryInputNCHW = batch_binaryInputNCHW,
                    globalInputNC = batch_globalInputNC,
                    policyTargetsNCMove = batch_policyTargetsNCMove,
                    globalTargetsNC = batch_globalTargetsNC,
                    scoreDistrN = batch_scoreDistrN,
                    valueTargetsNCHW = batch_valueTargetsNCHW,
                )
                if include_meta:
                    batch["metadataInputNC"] = batch_metadataInputNC
                if include_qvalues and qValueTargetsNCMove is not None:
                    batch["qValueTargetsNCMove"] = batch_qValueTargetsNCMove
                if include_weighted_values and weightedValueTargetsNC is not None:
                    batch["weightedValueTargetsNC"] = batch_weightedValueTargetsNC
                if include_connection_targets and connectionTargetsNPP is not None:
                    batch["connectionTargetsNPP"] = batch_connectionTargetsNPP

                yield batch


def apply_symmetry_policy(tensor, symm, pos_len):
    """Same as apply_symmetry but also handles the pass index"""
    batch_size = tensor.shape[0]
    channels = tensor.shape[1]
    tensor_without_pass = tensor[:,:,:-1].view((batch_size, channels, pos_len, pos_len))
    tensor_transformed = apply_symmetry(tensor_without_pass, symm)
    return torch.cat((
        tensor_transformed.reshape(batch_size, channels, pos_len*pos_len),
        tensor[:,:,-1:]
    ), dim=2)

def apply_symmetry(tensor, symm):
    """
    Apply a symmetry operation to the given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be rotated. (..., W, W)
        symm (int):
            0, 1, 2, 3: Rotation by symm * pi / 2 radians.
            4, 5, 6, 7: Mirror symmetry on top of rotation.
    """
    assert tensor.shape[-1] == tensor.shape[-2]

    if symm == 0:
        return tensor
    if symm == 1:
        return tensor.transpose(-2, -1).flip(-2)
    if symm == 2:
        return tensor.flip(-1).flip(-2)
    if symm == 3:
        return tensor.transpose(-2, -1).flip(-1)
    if symm == 4:
        return tensor.transpose(-2, -1)
    if symm == 5:
        return tensor.flip(-1)
    if symm == 6:
        return tensor.transpose(-2, -1).flip(-1).flip(-2)
    if symm == 7:
        return tensor.flip(-2)


def build_history_matrices(model_config: modelconfigs.ModelConfig, device):
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    # Accept 22 (older) or 23 (with weight mask) features
    assert num_bin_features in (22,23), f"Unexpected num_bin_features={num_bin_features}"

    base_vals = [
        1.0,  # 0
        1.0,  # 1
        1.0,  # 2
        1.0,  # 3
        1.0,  # 4
        1.0,  # 5
        1.0,  # 6
        1.0,  # 7
        1.0,  # 8
        0.0,  # 9   Location of move 1 turn ago
        0.0,  # 10  Location of move 2 turns ago
        0.0,  # 11  Location of move 3 turns ago
        0.0,  # 12  Location of move 4 turns ago
        0.0,  # 13  Location of move 5 turns ago
        1.0,  # 14  Ladder-threatened stone
        0.0,  # 15  Ladder-threatened stone, 1 turn ago
        0.0,  # 16  Ladder-threatened stone, 2 turns ago
        1.0,  # 17
        1.0,  # 18
        1.0,  # 19
        1.0,  # 20
        1.0,  # 21
    ]
    if num_bin_features == 23:
        base_vals.append(1.0)  # 22 weight mask feature - static (not history-shifted)
    h_base = torch.diag(torch.tensor(base_vals, device=device, requires_grad=False))
    # Because we have ladder features that express past states rather than past diffs,
    # the most natural encoding when we have no history is that they were always the
    # same, rather than that they were all zero. So rather than zeroing them we have no
    # history, we add entries in the matrix to copy them over.
    # By default, without history, the ladder features 15 and 16 just copy over from 14.
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    h0 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prev move, we enable feature 9 and 15
    h0[9, 9] = 1.0  # Enable 9 -> 9
    h0[14, 15] = -1.0  # Stop copying 14 -> 15
    h0[14, 16] = -1.0  # Stop copying 14 -> 16
    h0[15, 15] = 1.0  # Enable 15 -> 15
    h0[15, 16] = 1.0  # Start copying 15 -> 16

    h1 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    # When have the prevprev move, we enable feature 10 and 16
    h1[10, 10] = 1.0  # Enable 10 -> 10
    h1[15, 16] = -1.0  # Stop copying 15 -> 16
    h1[16, 16] = 1.0  # Enable 16 -> 16

    h2 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h2[11, 11] = 1.0

    h3 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h3[12, 12] = 1.0

    h4 = torch.zeros(num_bin_features, num_bin_features, device=device, requires_grad=False)
    h4[13, 13] = 1.0

    # (1, n_bin, n_bin)
    h_base = h_base.reshape((1, num_bin_features, num_bin_features))
    # (5, n_bin, n_bin)
    h_builder = torch.stack((h0, h1, h2, h3, h4), dim=0)

    return (h_base, h_builder)


def apply_history_matrices(model_config, batch_binaryInputNCHW, batch_globalInputNC, batch_globalTargetsNC, h_base, h_builder):
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    # include_history = batch_globalTargetsNC[:,36:41]
    should_stop_history = torch.rand_like(batch_globalTargetsNC[:,36:41]) >= 0.98
    include_history = (torch.cumsum(should_stop_history,axis=1,dtype=torch.float32) <= 0.1).to(torch.float32)

    # include_history: (N, 5)
    # bi * ijk -> bjk, (N, 5) * (5, n_bin, n_bin) -> (N, n_bin, n_bin)
    h_matrix = h_base + torch.einsum("bi,ijk->bjk", include_history, h_builder)


    # batch_binaryInputNCHW: (N, n_bin_in, 19, 19)
    # h_matrix: (N, n_bin_in, n_bin_out)
    # Result: (N, n_bin_out, 19, 19)
    batch_binaryInputNCHW = torch.einsum("bijk,bil->bljk", batch_binaryInputNCHW, h_matrix)

    # First 5 global input features exactly correspond to include_history, pointwise multiply to
    # enable/disable them
    batch_globalInputNC = batch_globalInputNC * torch.nn.functional.pad(
        include_history, ((0, num_global_features - include_history.shape[1])), value=1.0
    )
    return batch_binaryInputNCHW, batch_globalInputNC
