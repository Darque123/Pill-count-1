//
//  CountSmoother.swift
//  PillCount
//
//  Temporal anti-flicker: the on-screen count only "locks" after the raw
//  per-frame count has been identical for `lockFrames` consecutive frames,
//  and an established lock is only abandoned after `breakFrames` consecutive
//  disagreeing frames. In between, the UI shows "counting…" so the user
//  knows not to trust the number yet.
//

import Foundation

struct CountSmoother {
    /// Consecutive identical frame counts required to enter `.locked`.
    /// At ~15 processed fps, 8 frames ≈ 0.5 s — inside the ~1 s target.
    var lockFrames: Int = 8
    /// Consecutive frames disagreeing with the locked value before the lock
    /// is dropped back to `.counting` (absorbs single-frame flicker).
    var breakFrames: Int = 3

    private(set) var state: CountState = .searching
    private var runValue: Int = -1
    private var runLength: Int = 0
    private var lockedValue: Int?

    /// Feed one raw frame count; returns the new user-facing state.
    @discardableResult
    mutating func ingest(_ count: Int) -> CountState {
        if count == runValue {
            runLength += 1
        } else {
            runValue = count
            runLength = 1
        }

        if runLength >= lockFrames {
            lockedValue = runValue
        } else if let locked = lockedValue, runValue != locked,
                  runLength >= breakFrames {
            lockedValue = nil
        }

        if let locked = lockedValue {
            state = .locked(locked)
        } else {
            state = .counting(runValue)
        }
        return state
    }

    mutating func reset() {
        state = .searching
        runValue = -1
        runLength = 0
        lockedValue = nil
    }
}
