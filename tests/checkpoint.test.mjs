// Unit tests for render-checkpoint.js (Fix 3) against fake-indexeddb — the
// one new dependency in this repo's now-existing package.json, since Node
// has no native IndexedDB. Each test uses a unique jobId so they can share
// the same in-memory fake database without resetting it between cases.
import 'fake-indexeddb/auto';
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    saveJobMeta,
    saveChunk,
    loadJobChunks,
    listIncompleteJobs,
    markJobComplete,
    deleteJob,
} from '../render-checkpoint.js';

function uniqueJobId(label) {
    return `test-${label}-${Math.random().toString(36).slice(2)}`;
}

test('saveJobMeta + listIncompleteJobs: a saved job appears until completed', async () => {
    const jobId = uniqueJobId('meta');
    await saveJobMeta(jobId, { fileName: 'clip.mp4', fileSize: 1024, totalFrames: 100 });

    const incomplete = await listIncompleteJobs();
    const found = incomplete.find((j) => j.jobId === jobId);
    assert.ok(found, 'job should be listed as incomplete right after saving');
    assert.equal(found.fileName, 'clip.mp4');
    assert.equal(found.completedAt, null);

    await deleteJob(jobId);
});

test('saveChunk + loadJobChunks: round-trips into the chunkMp4Maps shape script.js expects', async () => {
    const jobId = uniqueJobId('chunks');
    const bytesA = new Uint8Array([1, 2, 3]);
    const bytesB = new Uint8Array([4, 5, 6]);

    await saveChunk(jobId, 0, 0, bytesA);
    await saveChunk(jobId, 0, 1, bytesB);
    await saveChunk(jobId, 1, 0, bytesA);

    const chunks = await loadJobChunks(jobId);
    assert.deepEqual(Object.keys(chunks).sort(), ['0', '1']);
    assert.deepEqual([...chunks[0][0]], [1, 2, 3]);
    assert.deepEqual([...chunks[0][1]], [4, 5, 6]);
    assert.deepEqual([...chunks[1][0]], [1, 2, 3]);

    await deleteJob(jobId);
});

test('markJobComplete: removes the job from listIncompleteJobs without deleting it', async () => {
    const jobId = uniqueJobId('complete');
    await saveJobMeta(jobId, { fileName: 'done.mp4' });
    await markJobComplete(jobId);

    const incomplete = await listIncompleteJobs();
    assert.ok(!incomplete.some((j) => j.jobId === jobId), 'completed job should not be listed as incomplete');

    await deleteJob(jobId);
});

test('deleteJob: removes both the job metadata and all of its chunks', async () => {
    const jobId = uniqueJobId('delete');
    await saveJobMeta(jobId, { fileName: 'gone.mp4' });
    await saveChunk(jobId, 0, 0, new Uint8Array([9]));
    await saveChunk(jobId, 0, 1, new Uint8Array([9]));

    await deleteJob(jobId);

    const incomplete = await listIncompleteJobs();
    assert.ok(!incomplete.some((j) => j.jobId === jobId), 'job metadata should be gone');

    const chunksAfter = await loadJobChunks(jobId);
    assert.deepEqual(chunksAfter, {}, 'no chunks should remain for a deleted job');
});

test('loadJobChunks on an unknown job returns an empty map, not an error', async () => {
    const chunks = await loadJobChunks(uniqueJobId('never-existed'));
    assert.deepEqual(chunks, {});
});
