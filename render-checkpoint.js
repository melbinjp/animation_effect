// Durable checkpoint storage for in-progress video renders (Fix 3): persists
// each completed chunk's bytes to IndexedDB as it finishes, so a killed tab
// or an accidental reload doesn't lose a multi-hour render — only the frames
// since the last completed chunk. Independent of the render pipeline's
// internals; the pipeline only ever calls the functions exported here.
//
// IndexedDB gotcha this file is written around: you cannot `await` between
// two requests on the same transaction and reliably keep using it — some
// browsers auto-commit the transaction while the await is pending. Every
// function here either issues all its requests before any await, or splits
// a read-then-write into two separate transactions.
'use strict';

const DB_NAME = 'linearty-render-checkpoints';
const DB_VERSION = 1;
const JOBS_STORE = 'jobs';
const CHUNKS_STORE = 'chunks';

function openCheckpointDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(JOBS_STORE)) {
                db.createObjectStore(JOBS_STORE, { keyPath: 'jobId' });
            }
            if (!db.objectStoreNames.contains(CHUNKS_STORE)) {
                const store = db.createObjectStore(CHUNKS_STORE, { keyPath: 'key' });
                store.createIndex('jobId', 'jobId', { unique: false });
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

function promisifyRequest(req) {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

function promisifyTx(tx) {
    return new Promise((resolve, reject) => {
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
    });
}

async function saveJobMeta(jobId, meta) {
    const db = await openCheckpointDB();
    const tx = db.transaction(JOBS_STORE, 'readwrite');
    tx.objectStore(JOBS_STORE).put({
        jobId,
        ...meta,
        createdAt: meta.createdAt || Date.now(),
        completedAt: null,
    });
    await promisifyTx(tx);
    db.close();
}

async function saveChunk(jobId, segIdx, chunkIdx, mp4Bytes) {
    const db = await openCheckpointDB();
    const tx = db.transaction(CHUNKS_STORE, 'readwrite');
    tx.objectStore(CHUNKS_STORE).put({
        key: `${jobId}/${segIdx}/${chunkIdx}`,
        jobId,
        segIdx,
        chunkIdx,
        bytes: mp4Bytes,
    });
    await promisifyTx(tx);
    db.close();
}

// Returns { [segIdx]: { [chunkIdx]: Uint8Array } } — the same shape
// chunkMp4Maps uses in script.js, so a resume can drop it straight in.
async function loadJobChunks(jobId) {
    const db = await openCheckpointDB();
    const tx = db.transaction(CHUNKS_STORE, 'readonly');
    const index = tx.objectStore(CHUNKS_STORE).index('jobId');
    const rows = await promisifyRequest(index.getAll(IDBKeyRange.only(jobId)));
    await promisifyTx(tx);
    db.close();

    const map = {};
    for (const row of rows) {
        if (!map[row.segIdx]) map[row.segIdx] = {};
        map[row.segIdx][row.chunkIdx] = row.bytes;
    }
    return map;
}

async function listIncompleteJobs() {
    const db = await openCheckpointDB();
    const tx = db.transaction(JOBS_STORE, 'readonly');
    const all = await promisifyRequest(tx.objectStore(JOBS_STORE).getAll());
    await promisifyTx(tx);
    db.close();
    return all.filter((job) => !job.completedAt);
}

async function markJobComplete(jobId) {
    const db = await openCheckpointDB();

    const readTx = db.transaction(JOBS_STORE, 'readonly');
    const job = await promisifyRequest(readTx.objectStore(JOBS_STORE).get(jobId));
    await promisifyTx(readTx);

    if (job) {
        job.completedAt = Date.now();
        const writeTx = db.transaction(JOBS_STORE, 'readwrite');
        writeTx.objectStore(JOBS_STORE).put(job);
        await promisifyTx(writeTx);
    }
    db.close();
}

async function deleteJob(jobId) {
    const db = await openCheckpointDB();

    const readTx = db.transaction(CHUNKS_STORE, 'readonly');
    const chunkIndex = readTx.objectStore(CHUNKS_STORE).index('jobId');
    const keys = await promisifyRequest(chunkIndex.getAllKeys(IDBKeyRange.only(jobId)));
    await promisifyTx(readTx);

    const writeTx = db.transaction([JOBS_STORE, CHUNKS_STORE], 'readwrite');
    writeTx.objectStore(JOBS_STORE).delete(jobId);
    const chunkStore = writeTx.objectStore(CHUNKS_STORE);
    for (const key of keys) chunkStore.delete(key);
    await promisifyTx(writeTx);
    db.close();
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        saveJobMeta,
        saveChunk,
        loadJobChunks,
        listIncompleteJobs,
        markJobComplete,
        deleteJob,
        DB_NAME,
        DB_VERSION,
    };
}
