// Node's built-in test runner (no new dependency — this repo has no
// package.json/npm setup at all, so none is being introduced for this) for
// the pure logic backing Fix 1 (retry), Fix 2 (partial-render salvage), and
// Fix 3 (resume) in ../script.js. All four functions under test live in
// ../render-math.js specifically because they're DOM-free and can be
// required from plain Node — script.js's top level assumes a browser and
// cannot be.
//
// render-checkpoint.js (the actual IndexedDB code, Fix 3) is deliberately
// not unit tested here: faking IndexedDB well enough to trust the result
// would need a new dependency in a repo that has none. It's covered by the
// manual resume check in the plan's Verification section instead.

import test from 'node:test';
import assert from 'node:assert/strict';
import {
    computeExpectedChunksForSegment,
    longestCompleteSegmentPrefix,
    framesToSkipForResume,
    retryWithBackoff,
} from '../render-math.js';

test('computeExpectedChunksForSegment: even division, one chunk per segment worth of frames', () => {
    // 300 frames, 3 segments of 100 each, chunk size 100 -> 1 chunk/segment.
    assert.equal(computeExpectedChunksForSegment(0, 3, 300, 100, 100), 1);
    assert.equal(computeExpectedChunksForSegment(2, 3, 300, 100, 100), 1);
});

test('computeExpectedChunksForSegment: multiple chunks per segment', () => {
    // 100 frames/segment, chunk size 30 -> ceil(100/30) = 4 chunks.
    assert.equal(computeExpectedChunksForSegment(0, 3, 300, 100, 30), 4);
});

test('computeExpectedChunksForSegment: final segment absorbs the remainder (the 52ff1cc case)', () => {
    // 310 frames, 3 segments, framesPerSegment = ceil(310/3) = 104.
    // Segment 2 (last) gets 310 - 2*104 = 102 frames, not 104.
    const framesPerSegment = 104;
    const totalFrames = 310;
    // Non-final segments: full framesPerSegment, chunk 40 -> ceil(104/40) = 3.
    assert.equal(computeExpectedChunksForSegment(0, 3, totalFrames, framesPerSegment, 40), 3);
    assert.equal(computeExpectedChunksForSegment(1, 3, totalFrames, framesPerSegment, 40), 3);
    // Final segment: 102 frames, chunk 40 -> ceil(102/40) = 3 too, but from a
    // different frame count — assert the frame count math directly matters
    // by using a chunk size where it would differ (e.g. 51).
    assert.equal(computeExpectedChunksForSegment(2, 3, totalFrames, framesPerSegment, 51), 2); // ceil(102/51)
    assert.equal(computeExpectedChunksForSegment(0, 3, totalFrames, framesPerSegment, 51), 3); // ceil(104/51)
});

test('longestCompleteSegmentPrefix: all segments complete', () => {
    const chunkMp4Maps = [{ 0: 'a' }, { 0: 'b' }, { 0: 'c' }];
    const expected = [1, 1, 1];
    assert.deepEqual(longestCompleteSegmentPrefix(chunkMp4Maps, expected), [0, 1, 2]);
});

test('longestCompleteSegmentPrefix: a gap partway through stops the prefix there', () => {
    const chunkMp4Maps = [{ 0: 'a' }, {}, { 0: 'c' }]; // segment 1 missing its chunk
    const expected = [1, 1, 1];
    assert.deepEqual(longestCompleteSegmentPrefix(chunkMp4Maps, expected), [0]);
});

test('longestCompleteSegmentPrefix: segment 0 itself incomplete salvages nothing', () => {
    const chunkMp4Maps = [{}, { 0: 'b' }, { 0: 'c' }];
    const expected = [1, 1, 1];
    assert.deepEqual(longestCompleteSegmentPrefix(chunkMp4Maps, expected), []);
});

test('longestCompleteSegmentPrefix: partial chunk count within a segment counts as incomplete', () => {
    // Segment 1 expects 2 chunks but only has 1 — not complete, even though
    // segment 0 is fine and segment 2 (never reached) is irrelevant.
    const chunkMp4Maps = [{ 0: 'a' }, { 0: 'b' }, { 0: 'c', 1: 'd' }];
    const expected = [1, 2, 2];
    assert.deepEqual(longestCompleteSegmentPrefix(chunkMp4Maps, expected), [0]);
});

test('framesToSkipForResume: matches the frame loop\'s own segIdx/localIdx math', () => {
    // 10 frames, 2 segments of 5, chunk size 2 -> segment 0 has chunks {0:[0,1],1:[2,3],2:[4]},
    // segment 1 (frames 5-9) has chunks {0:[5,6],1:[7,8],2:[9]}.
    const completed = ['0-0', '0-1', '1-2']; // seg0 chunks 0&1 done, seg1's last chunk done
    const skip = framesToSkipForResume(10, 2, 5, 2, completed);
    assert.deepEqual([...skip].sort((a, b) => a - b), [0, 1, 2, 3, 9]);
});

test('framesToSkipForResume: nothing completed skips nothing', () => {
    const skip = framesToSkipForResume(10, 2, 5, 2, []);
    assert.equal(skip.size, 0);
});

test('retryWithBackoff: succeeds without retrying when the first attempt works', async () => {
    let calls = 0;
    const result = await retryWithBackoff(async () => { calls++; return 'ok'; }, 3, 0);
    assert.equal(result, 'ok');
    assert.equal(calls, 1);
});

test('retryWithBackoff: retries a failing attempt and succeeds within the budget', async () => {
    let calls = 0;
    const result = await retryWithBackoff(async (attempt) => {
        calls++;
        if (attempt < 3) throw new Error(`fail on attempt ${attempt}`);
        return 'recovered';
    }, 3, 0);
    assert.equal(result, 'recovered');
    assert.equal(calls, 3);
});

test('retryWithBackoff: exhausts attempts and throws the last error', async () => {
    let calls = 0;
    await assert.rejects(
        retryWithBackoff(async () => { calls++; throw new Error('always fails'); }, 3, 0),
        /always fails/
    );
    assert.equal(calls, 3);
});
