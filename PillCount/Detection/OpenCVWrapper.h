//
//  OpenCVWrapper.h
//  PillCount
//
//  Objective-C interface to the C++ detection pipeline (PillPipeline.cpp),
//  consumable from Swift via the bridging header. This header is pure
//  Objective-C — all C++ stays inside the .mm file.
//

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

/// Tunable detection parameters. Defaults mirror pillcount::Params in
/// PillPipeline.hpp — see that file for per-parameter documentation.
@interface PCDetectorParams : NSObject
@property (nonatomic) int maxDimension;
@property (nonatomic) double bgSmoothFraction;
@property (nonatomic) int blurKernel;
@property (nonatomic) int minDeviation;
@property (nonatomic) int minSaturation;
@property (nonatomic) int minGradient;
@property (nonatomic) int gradCloseKernel;
@property (nonatomic) double minAreaFraction;
@property (nonatomic) double maxAreaFraction;
@property (nonatomic) double minSolidity;
@property (nonatomic) double maxAspect;
@property (nonatomic) double distRatio;
@property (nonatomic) double markerWindowFraction;
@property (nonatomic) BOOL valleyAssist;
@property (nonatomic) double valleyRatio;
@property (nonatomic) int valleyFloor;
@property (nonatomic) int valleyKernel;
@property (nonatomic) int minBoundaryGrad;
@property (nonatomic) int openKernel;
@property (nonatomic) int closeKernel;
@end

/// One detected pill. Coordinates are normalized to [0, 1] in the frame's
/// (post-rotation) pixel space, so they can be mapped onto any view size.
@interface PCDetectedPill : NSObject
@property (nonatomic, readonly) NSArray<NSValue *> *contour;  // boxed CGPoint
@property (nonatomic, readonly) CGPoint center;
@property (nonatomic, readonly) double areaFraction;
@end

@interface PCDetectionFrameResult : NSObject
@property (nonatomic, readonly) NSArray<PCDetectedPill *> *pills;
@property (nonatomic, readonly) NSInteger count;
@property (nonatomic, readonly) double processingMillis;
@end

/// Thin, thread-safe wrapper around pillcount::detectPills().
@interface PCPillDetector : NSObject
- (instancetype)initWithParams:(PCDetectorParams *)params;
/// Detect pills in a BGRA pixel buffer (the app's video output format).
/// Returns an empty result for unsupported pixel formats.
- (PCDetectionFrameResult *)detectInPixelBuffer:(CVPixelBufferRef)pixelBuffer
    NS_SWIFT_NAME(detect(in:));
@end

NS_ASSUME_NONNULL_END
