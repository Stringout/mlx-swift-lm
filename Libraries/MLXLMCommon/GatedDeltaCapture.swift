// Copyright © 2026 Apple Inc.

// Gated-delta capture and masked replay for DFlash2 verify passes, ported from
// ml-explore/mlx-swift-lm#607 (spokvulcan). The reference rolls back the same way
// (`_GDNStateCapture.rollback` in z-lab/dflash `dflash/model_mlx.py`).

import Foundation
import MLX

/// The gated-delta scan of a verify pass.
///
/// `gatedDeltaUpdate` sends an unmasked call with a supported head layout to
/// `MLXFast.gatedDeltaUpdate` and a masked call to the package's Metal kernel
/// (or its ops fallback). Replay must mask, so the verify pass masks too, with
/// every position valid: both then take one route, and a replayed state is the
/// state this scan held after that many positions.
public func gatedDeltaVerifyUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    aLog: MLXArray, dtBias: MLXArray, state: MLXArray?, useKernel: Bool = true
) -> (MLXArray, MLXArray) {
    let mask = MLXArray.ones([q.dim(0), q.dim(1)], dtype: .bool)
    return gatedDeltaUpdate(
        q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state, mask: mask,
        useKernel: useKernel)
}

/// What one gated-delta layer consumed in a verify pass: enough to recompute its
/// state as if only a prefix of the pass had run.
public struct GatedDeltaCapture {
    /// `concat([convState, qkv])`, `[1, K - 1 + S, convDim]`.
    public var convInput: MLXArray
    /// Normalized, scaled `q`/`k` (`[1, S, Hk, Dk]`) and `v` (`[1, S, Hv, Dv]`).
    public var q: MLXArray
    public var k: MLXArray
    public var v: MLXArray
    /// Raw gate projections, `[1, S, Hv]`.
    public var a: MLXArray
    public var b: MLXArray
    public var aLog: MLXArray
    public var dtBias: MLXArray
    /// Recurrent state before the pass.
    public var initialState: MLXArray

    public init(
        convInput: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray,
        a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
        initialState: MLXArray
    ) {
        self.convInput = convInput
        self.q = q
        self.k = k
        self.v = v
        self.a = a
        self.b = b
        self.aLog = aLog
        self.dtBias = dtBias
        self.initialState = initialState
    }

    /// The per-pass arrays, so a compiled verify body can return them as outputs.
    package var arrays: [MLXArray] { [convInput, q, k, v, a, b] }

    package init(arrays: [MLXArray], aLog: MLXArray, dtBias: MLXArray, initialState: MLXArray) {
        precondition(arrays.count == 6, "convInput, q, k, v, a, b")
        self.init(
            convInput: arrays[0], q: arrays[1], k: arrays[2], v: arrays[3], a: arrays[4],
            b: arrays[5], aLog: aLog, dtBias: dtBias, initialState: initialState)
    }

    /// The layer's recurrent and conv state after the first `validCount` positions.
    ///
    /// Positions past `validCount` are masked out of the scan, so the result is
    /// exact for any count and `validCount` may be a lazy `[]` int32 array.
    public func replay(validCount: MLXArray) -> (recurrent: MLXArray, conv: MLXArray) {
        let outputs = compiledGatedDeltaReplay([
            q, k, v, a, b, aLog, dtBias, initialState, convInput, validCount,
        ])
        return (outputs[0], outputs[1])
    }
}

/// One trace serves every layer: their shapes match.
private let compiledGatedDeltaReplay: @Sendable ([MLXArray]) -> [MLXArray] = compile { inputs in
    let (q, k, v) = (inputs[0], inputs[1], inputs[2])
    let (a, b, aLog, dtBias) = (inputs[3], inputs[4], inputs[5], inputs[6])
    let (state, convInput, validCount) = (inputs[7], inputs[8], inputs[9])
    let s = q.dim(1)
    let mask = (MLXArray(Int32(0) ..< Int32(s)) .< validCount.asType(.int32))
        .expandedDimensions(axis: 0)
    let (_, newState) = gatedDeltaUpdate(
        q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state, mask: mask)
    // The conv state after `validCount` positions: rows [validCount, validCount + K - 1).
    let kernelRows = convInput.dim(1) - s
    let rows = (validCount.asType(.int32) + MLXArray(Int32(0) ..< Int32(kernelRows)))
        .reshaped([1, kernelRows, 1])
    let conv = contiguous(takeAlong(convInput, rows, axis: 1))
    return [newState, conv]
}
