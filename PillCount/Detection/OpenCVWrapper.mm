//
//  OpenCVWrapper.mm
//  PillCount
//
//  Adapter between CoreVideo/Swift and the pure C++ pipeline: converts the
//  camera's BGRA pixel buffer to a cv::Mat, runs pillcount::detectPills, and
//  boxes the results into Objective-C objects for Swift.
//

// OpenCV must be included before any Apple header (macro collisions).
#import <opencv2/core.hpp>
#import <opencv2/imgproc.hpp>

#import "PillPipeline.hpp"

#import "OpenCVWrapper.h"
#import <UIKit/UIKit.h>

#include <chrono>

@implementation PCDetectorParams

- (instancetype)init {
    if ((self = [super init])) {
        // Single source of truth for defaults: the C++ Params struct.
        pillcount::Params d;
        _maxDimension = d.maxDimension;
        _bgBlurFraction = d.bgBlurFraction;
        _blurKernel = d.blurKernel;
        _minDeviation = d.minDeviation;
        _minSaturation = d.minSaturation;
        _minGradient = d.minGradient;
        _gradCloseKernel = d.gradCloseKernel;
        _minAreaFraction = d.minAreaFraction;
        _maxAreaFraction = d.maxAreaFraction;
        _minSolidity = d.minSolidity;
        _maxAspect = d.maxAspect;
        _distRatio = d.distRatio;
        _markerWindowFraction = d.markerWindowFraction;
        _valleyAssist = d.valleyAssist;
        _valleyThreshold = d.valleyThreshold;
        _valleyKernel = d.valleyKernel;
        _openKernel = d.openKernel;
        _closeKernel = d.closeKernel;
    }
    return self;
}

- (pillcount::Params)cppParams {
    pillcount::Params p;
    p.maxDimension = self.maxDimension;
    p.bgBlurFraction = self.bgBlurFraction;
    p.blurKernel = self.blurKernel;
    p.minDeviation = self.minDeviation;
    p.minSaturation = self.minSaturation;
    p.minGradient = self.minGradient;
    p.gradCloseKernel = self.gradCloseKernel;
    p.minAreaFraction = self.minAreaFraction;
    p.maxAreaFraction = self.maxAreaFraction;
    p.minSolidity = self.minSolidity;
    p.maxAspect = self.maxAspect;
    p.distRatio = self.distRatio;
    p.markerWindowFraction = self.markerWindowFraction;
    p.valleyAssist = self.valleyAssist;
    p.valleyThreshold = self.valleyThreshold;
    p.valleyKernel = self.valleyKernel;
    p.openKernel = self.openKernel;
    p.closeKernel = self.closeKernel;
    return p;
}

@end

@implementation PCDetectedPill {
    NSArray<NSValue *> *_contour;
    CGPoint _center;
    double _areaFraction;
}

- (instancetype)initWithPill:(const pillcount::Pill &)pill {
    if ((self = [super init])) {
        NSMutableArray<NSValue *> *points =
            [NSMutableArray arrayWithCapacity:pill.contour.size()];
        for (const cv::Point2f &pt : pill.contour) {
            [points addObject:[NSValue valueWithCGPoint:CGPointMake(pt.x, pt.y)]];
        }
        _contour = points;
        _center = CGPointMake(pill.center.x, pill.center.y);
        _areaFraction = pill.areaFraction;
    }
    return self;
}

- (NSArray<NSValue *> *)contour { return _contour; }
- (CGPoint)center { return _center; }
- (double)areaFraction { return _areaFraction; }

@end

@implementation PCDetectionFrameResult {
    NSArray<PCDetectedPill *> *_pills;
    double _processingMillis;
}

- (instancetype)initWithPills:(NSArray<PCDetectedPill *> *)pills
             processingMillis:(double)millis {
    if ((self = [super init])) {
        _pills = pills;
        _processingMillis = millis;
    }
    return self;
}

- (NSArray<PCDetectedPill *> *)pills { return _pills; }
- (NSInteger)count { return (NSInteger)_pills.count; }
- (double)processingMillis { return _processingMillis; }

@end

@implementation PCPillDetector {
    pillcount::Params _params;
}

- (instancetype)initWithParams:(PCDetectorParams *)params {
    if ((self = [super init])) {
        _params = [params cppParams];
    }
    return self;
}

- (PCDetectionFrameResult *)detectInPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    auto start = std::chrono::steady_clock::now();

    if (CVPixelBufferGetPixelFormatType(pixelBuffer) !=
        kCVPixelFormatType_32BGRA) {
        return [[PCDetectionFrameResult alloc] initWithPills:@[]
                                            processingMillis:0];
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    void *base = CVPixelBufferGetBaseAddress(pixelBuffer);
    size_t width = CVPixelBufferGetWidth(pixelBuffer);
    size_t height = CVPixelBufferGetHeight(pixelBuffer);
    size_t stride = CVPixelBufferGetBytesPerRow(pixelBuffer);

    pillcount::FrameResult result;
    if (base != nullptr) {
        // Zero-copy view of the buffer; cvtColor makes the working copy.
        cv::Mat bgra((int)height, (int)width, CV_8UC4, base, stride);
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        result = pillcount::detectPills(bgr, _params);
    } else {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    }

    NSMutableArray<PCDetectedPill *> *pills =
        [NSMutableArray arrayWithCapacity:result.pills.size()];
    for (const pillcount::Pill &pill : result.pills) {
        [pills addObject:[[PCDetectedPill alloc] initWithPill:pill]];
    }

    double millis = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    return [[PCDetectionFrameResult alloc] initWithPills:pills
                                        processingMillis:millis];
}

@end
