//
//  DetectionModels.swift
//  PillCount
//
//  Swift-side value types for detection results and count state.
//

import CoreGraphics
import Foundation

/// One detected pill, in normalized [0, 1] frame coordinates.
struct DetectedPill: Identifiable {
    let id: Int
    let contour: [CGPoint]
    let center: CGPoint
}

/// The result of processing one camera frame.
struct DetectionFrame {
    let pills: [DetectedPill]
    /// Pixel size of the (post-rotation) camera frame; needed to map
    /// normalized contours onto the aspect-fill preview.
    let imageSize: CGSize
    let processingMillis: Double
    let timestamp: Date

    var count: Int { pills.count }
}

/// The user-facing counting state produced by `CountSmoother`.
enum CountState: Equatable {
    /// No frames processed yet (camera starting, or permission missing).
    case searching
    /// A count is displayed but has not been stable long enough to trust.
    case counting(Int)
    /// The count has been identical for `lockFrames` consecutive frames.
    case locked(Int)

    var displayCount: Int? {
        switch self {
        case .searching: return nil
        case .counting(let n), .locked(let n): return n
        }
    }

    var isLocked: Bool {
        if case .locked = self { return true }
        return false
    }
}
