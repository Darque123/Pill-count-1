//
//  CameraPreviewView.swift
//  PillCount
//
//  SwiftUI wrapper for AVCaptureVideoPreviewLayer with tap-to-focus.
//  Uses .resizeAspectFill — DetectionOverlayView applies the same aspect-fill
//  mapping so contours line up with the video.
//

import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let onTapFocus: (CGPoint) -> Void  // normalized capture-device point

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.onTapFocus = onTapFocus
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    final class PreviewUIView: UIView {
        var onTapFocus: ((CGPoint) -> Void)?

        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }

        override init(frame: CGRect) {
            super.init(frame: frame)
            addGestureRecognizer(UITapGestureRecognizer(
                target: self, action: #selector(handleTap(_:))))
        }

        required init?(coder: NSCoder) { fatalError("not used") }

        @objc private func handleTap(_ recognizer: UITapGestureRecognizer) {
            let layerPoint = recognizer.location(in: self)
            let devicePoint = previewLayer.captureDevicePointConverted(
                fromLayerPoint: layerPoint)
            onTapFocus?(devicePoint)
        }
    }
}
