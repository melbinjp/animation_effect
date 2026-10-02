'use strict';
// Dedicated CPU-only worker (forces OpenCV.js without attempting WebGPU).
// Reuses the canonical CpuProcessor implementation in gpu-worker.js to eliminate code duplication.
self.__FORCE_CPU__ = true;
importScripts('gpu-worker.js');
