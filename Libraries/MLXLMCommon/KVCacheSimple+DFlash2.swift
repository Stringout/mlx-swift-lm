// Copyright © 2026 Apple Inc.

// Attention rows for a DFlash2 verify pass, ported from ml-explore/mlx-swift-lm#607
// (spokvulcan). The pass writes at a position that may still be lazy and commits
// once the accept count is known.

import Foundation
import MLX

extension KVCacheSimple {
    /// Write `S` rows at `position` (`[1]` int32, possibly lazy) without moving
    /// `offset`, and return the first `visibleLength` rows. Rows past the
    /// committed offset are scratch for a later write to overwrite.
    package func writeRows(
        keys newKeys: MLXArray, values newValues: MLXArray,
        position: MLXArray, visibleLength: Int
    ) -> (MLXArray, MLXArray) {
        if keys == nil || visibleLength > keys!.dim(2) {
            let capacity = (visibleLength + step - 1) / step * step
            let kShape = [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)]
            let vShape = [newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)]
            let grownKeys = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let grownValues = MLXArray.zeros(vShape, dtype: newValues.dtype)
            if let keys, let values {
                // Keep the whole buffer: scratch rows may belong to a pass in flight.
                self.keys = concatenated(
                    [keys, grownKeys[.ellipsis, keys.dim(2)..., 0...]], axis: 2)
                self.values = concatenated(
                    [values, grownValues[.ellipsis, values.dim(2)..., 0...]], axis: 2)
            } else {
                keys = grownKeys
                values = grownValues
            }
        }
        let rows = (position.asType(.int32) + MLXArray(Int32(0) ..< Int32(newKeys.dim(2))))
            .reshaped([1, 1, -1, 1])
        keys = putAlong(keys!, rows, values: newKeys, axis: 2)
        values = putAlong(values!, rows, values: newValues, axis: 2)
        return (
            keys![.ellipsis, ..<visibleLength, 0...],
            values![.ellipsis, ..<visibleLength, 0...]
        )
    }

    /// The cache now holds `count` positions written by ``writeRows(keys:values:position:visibleLength:)``.
    package func commitRows(count: Int) {
        offset = count
    }
}
