using UnityEngine;
using System.Runtime.InteropServices;

namespace Laplace3D {

    struct Cell {
        public bool isBoundary;
        public float potential;
        public Vector3 idx;
        public Vector3 pos;
    }

    public class Laplace : MonoBehaviour {

        Cell[] cells;

        #region GPU
        const int SIMULATION_BLOCK_SIZE = 32;
        int threadGroupSize;
        int bufferSize;
        public ComputeShader LaplaceCS;
        ComputeBuffer bufferRead, bufferWrite;
        #endregion GPU

        public int width = 30;
        public int height = 30;
        public int depth = 30;

        [Range(0f, 1f)] public float front_strength;
        [Range(0f, 1f)] public float back_strength;
        [Range(0f, 1f)] public float left_strength;
        [Range(0f, 1f)] public float right_strength;
        [Range(0f, 1f)] public float up_strength;
        [Range(0f, 1f)] public float down_strength;

        void Start() {

            cells = new Cell[width * height * depth];
            bufferSize = cells.Length;
            
            threadGroupSize = Mathf.CeilToInt(bufferSize / SIMULATION_BLOCK_SIZE) + 1;
            bufferRead = new ComputeBuffer(bufferSize, Marshal.SizeOf(typeof(Cell)));
            bufferWrite = new ComputeBuffer(bufferSize, Marshal.SizeOf(typeof(Cell)));

            SetBoundaryCondition();
        }

        void Update() {
            
            CalcLaplace();

        }

        void OnDestroy() {
            bufferRead.Release();
            bufferWrite.Release();
        }

        void SetBoundaryCondition() {
            
            for(int z = 0; z < depth; z++) {
                for(int y = 0; y < height; y++) {
                    for(int x = 0; x < width; x++) {
                        int idx = width * height * z + width * y + x;
                        cells[idx].pos = new Vector3(x/ (float)width, y/ (float)height, z/ (float)depth);
                        cells[idx].idx = new Vector3(x, y, z);

                        cells[idx].potential = 0.5f;
                        cells[idx].isBoundary = false;
                        // front
                        if (z == 0) {
                            cells[idx].potential = front_strength;
                            cells[idx].isBoundary = true;
                        }
                        // back
                        if (z == depth - 1) {
                            cells[idx].potential = back_strength;
                            cells[idx].isBoundary = true;
                        }
                        // left
                        if (x == 0) {
                            cells[idx].potential = left_strength;
                            cells[idx].isBoundary = true;
                        }
                        // right
                        if (x == width - 1) {
                            cells[idx].potential = right_strength;
                            cells[idx].isBoundary = true;
                        }
                        // down
                        if (y == 0) {
                            cells[idx].potential = down_strength;
                            cells[idx].isBoundary = true;
                        }
                        // up
                        if (y == height - 1) {
                            cells[idx].potential = up_strength;
                            cells[idx].isBoundary = true;
                        }
                    }
                }
            }

            bufferRead.SetData(cells);
            bufferWrite.SetData(cells);

        }

        void CalcLaplace() {
            
            LaplaceCS.SetInt("WIDTH", width);
            LaplaceCS.SetInt("HEIGHT", height);
            LaplaceCS.SetInt("DEPTH", depth);

            int kernel = LaplaceCS.FindKernel("Laplace3D");
            LaplaceCS.SetBuffer(kernel, "_Read", bufferRead);
            LaplaceCS.SetBuffer(kernel, "_Write", bufferWrite);

            LaplaceCS.Dispatch(kernel, threadGroupSize, 1, 1);

            SwapBuffer();

        }

        void SwapBuffer() {
            ComputeBuffer tmp = bufferWrite;
            bufferWrite = bufferRead;
            bufferRead = tmp;
        }

        public ComputeBuffer GetBuffer() {
            return bufferRead;
        }

        public int GetBufferSize() {
            return bufferSize;
        }
    }
}