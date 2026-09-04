// Pure, DOM-free math shared between script.js (loaded as a plain global
// script in the browser) and the Node test suite in tests/. Kept in its own
// file because script.js's top level assumes a DOM (document.getElementById
// calls run at module-eval time) and can't be required from plain Node.
'use strict';

// Segment s's true frame count, accounting for the final segment absorbing
// the remainder when totalFrames isn't evenly divisible by segCount — the
// same calculation the 52ff1cc bug fix established; kept in exactly one
// place so the partial-render salvage path can never drift from what the
// frame loop actually did.
function computeExpectedChunksForSegment(segIdx, segCount, totalFrames, framesPerSegment, chunkFrames) {
    const segFrames = (segIdx === segCount - 1)
        ? (totalFrames - segIdx * framesPerSegment)
        : framesPerSegment;
    return Math.max(1, Math.ceil(segFrames / chunkFrames));
}

// The longest run of segments, starting at segment 0, where every expected
// chunk for that segment is present in chunkMp4Maps. Segments are frame-
// ordered, so a gap at the start means nothing is salvageable — a video
// missing its first N seconds isn't a usable partial result, even if later
// segments completed fine.
function longestCompleteSegmentPrefix(chunkMp4Maps, expectedCounts) {
    const prefix = [];
    for (let s = 0; s < expectedCounts.length; s++) {
        const present = Object.keys(chunkMp4Maps[s] || {}).length;
        if (present !== expectedCounts[s]) break;
        prefix.push(s);
    }
    return prefix;
}

// Given a job's frame/segment/chunk layout and the set of chunks already
// persisted from a prior attempt (Fix 3 resume), which global frame indices
// can be skipped entirely — no decode, no OpenCV, no WASM FS write — because
// their chunk is already durably complete.
function framesToSkipForResume(totalFrames, segCount, framesPerSegment, chunkFrames, completedChunkKeys) {
    const completed = new Set(completedChunkKeys); // keys like `${segIdx}-${chunkIdx}`
    const skip = new Set();
    for (let globalIdx = 0; globalIdx < totalFrames; globalIdx++) {
        const segIdx = Math.min(segCount - 1, Math.floor(globalIdx / framesPerSegment));
        const localIdx = globalIdx - segIdx * framesPerSegment;
        const chunkIdx = Math.floor(localIdx / chunkFrames);
        if (completed.has(`${segIdx}-${chunkIdx}`)) {
            skip.add(globalIdx);
        }
    }
    return skip;
}

// Generic bounded retry with linear backoff, used by encodeOneChunk in
// script.js. Kept here (not inline there) so the retry-then-throw behavior
// is unit-testable against a fake fn, without a real FFmpeg instance.
// fn receives the 1-based attempt number and either resolves or throws.
async function retryWithBackoff(fn, maxAttempts, backoffMs) {
    let lastError;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn(attempt);
        } catch (err) {
            lastError = err;
            if (attempt < maxAttempts && backoffMs > 0) {
                await new Promise((resolve) => setTimeout(resolve, backoffMs * attempt));
            }
        }
    }
    throw lastError;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        computeExpectedChunksForSegment,
        longestCompleteSegmentPrefix,
        framesToSkipForResume,
        retryWithBackoff,
    };
}
