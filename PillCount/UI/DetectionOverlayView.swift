//
//  DetectionOverlayView.swift
//  PillCount
//
//  Draws every detected pill's contour over the camera preview so the
//  pharmacist can visually verify that each pill was counted exactly once.
//  This per-pill visual audit is a safety feature, not decoration.
//

import SwiftUI

struct DetectionOverlayView: View {
    let frame: DetectionFrame?
    let locked: Bool

    var body: some View {
        Canvas { context, size in
            guard let frame else { return }
            let transform = Self.aspectFillTransform(
                imageSize: frame.imageSize, viewSize: size)
            let color: Color = locked ? .green : .yellow

            for pill in frame.pills {
                guard pill.contour.count > 2 else { continue }
                var path = Path()
                path.move(to: map(pill.contour[0], frame, transform))
                for point in pill.contour.dropFirst() {
                    path.addLine(to: map(point, frame, transform))
                }
                path.closeSubpath()
                context.stroke(path, with: .color(color), lineWidth: 2.5)
                context.fill(path, with: .color(color.opacity(0.12)))

                // Center dot: visible even when outlines overlap densely.
                let c = map(pill.center, frame, transform)
                context.fill(
                    Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3,
                                           width: 6, height: 6)),
                    with: .color(color))
            }
        }
        .allowsHitTesting(false)
    }

    private func map(_ normalized: CGPoint, _ frame: DetectionFrame,
                     _ t: (scale: CGFloat, offset: CGPoint)) -> CGPoint {
        CGPoint(
            x: normalized.x * frame.imageSize.width * t.scale + t.offset.x,
            y: normalized.y * frame.imageSize.height * t.scale + t.offset.y)
    }

    /// The same mapping AVCaptureVideoPreviewLayer applies with
    /// .resizeAspectFill, so overlay and video stay registered.
    static func aspectFillTransform(imageSize: CGSize, viewSize: CGSize)
        -> (scale: CGFloat, offset: CGPoint)
    {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return (1, .zero)
        }
        let scale = max(viewSize.width / imageSize.width,
                        viewSize.height / imageSize.height)
        let offset = CGPoint(
            x: (viewSize.width - imageSize.width * scale) / 2,
            y: (viewSize.height - imageSize.height * scale) / 2)
        return (scale, offset)
    }
}
