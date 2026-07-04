//
//  CameraManager.swift
//  PillCount
//
//  Owns the AVCaptureSession: back wide-angle camera, 1080p, BGRA frames
//  rotated to portrait, delivered to `onFrame` on a dedicated video queue.
//  Processing never leaves the device.
//

import AVFoundation
import Foundation

final class CameraManager: NSObject, ObservableObject {
    enum AuthorizationState {
        case undetermined, authorized, denied
    }

    let session = AVCaptureSession()

    @Published private(set) var authorization: AuthorizationState = .undetermined
    @Published private(set) var isTorchOn = false

    /// Called with each captured pixel buffer, on the video queue.
    var onFrame: ((CVPixelBuffer) -> Void)?

    private let sessionQueue = DispatchQueue(label: "pillcount.session")
    private let videoQueue = DispatchQueue(label: "pillcount.video", qos: .userInitiated)
    private var device: AVCaptureDevice?
    private var configured = false

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            authorization = .authorized
            startSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.authorization = granted ? .authorized : .denied
                    if granted { self.startSession() }
                }
            }
        default:
            authorization = .denied
        }
    }

    func stop() {
        sessionQueue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    private func startSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.configured {
                self.configure()
                self.configured = true
            }
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    private func configure() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1920x1080

        guard let device = AVCaptureDevice.default(
                  .builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input)
        else { return }
        session.addInput(input)
        self.device = device

        // Continuous autofocus/exposure keeps the tray sharp as it moves.
        if (try? device.lockForConfiguration()) != nil {
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            device.unlockForConfiguration()
        }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_32BGRA
        ]
        // Detection drops frames while busy, so late frames are useless.
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { return }
        session.addOutput(output)

        // Deliver buffers already rotated to portrait so detection, overlay,
        // and preview all share one coordinate space.
        if let connection = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            } else if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
    }

    /// Tap-to-focus at a normalized device point (from the preview layer's
    /// `captureDevicePointConverted`).
    func focus(atDevicePoint point: CGPoint) {
        sessionQueue.async { [weak self] in
            guard let device = self?.device,
                  (try? device.lockForConfiguration()) != nil else { return }
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = point
                device.focusMode = .continuousAutoFocus
            }
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = point
                device.exposureMode = .continuousAutoExposure
            }
            device.unlockForConfiguration()
        }
    }

    /// Torch can rescue a glare-heavy or dim pharmacy counter.
    func toggleTorch() {
        sessionQueue.async { [weak self] in
            guard let self, let device = self.device, device.hasTorch,
                  (try? device.lockForConfiguration()) != nil else { return }
            let newValue = device.torchMode != .on
            device.torchMode = newValue ? .on : .off
            device.unlockForConfiguration()
            DispatchQueue.main.async { self.isTorchOn = newValue }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }
        onFrame?(pixelBuffer)
    }
}
