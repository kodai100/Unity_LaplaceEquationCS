using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LatticeDebug : MonoBehaviour {

    public enum Mode {
        Both, Phase1, Phase2
    }
    public Mode mode = Mode.Both;

    #region GPU
    const int SIMULATION_BLOCK_SIZE = 256;
    int threadGroupSize;
    int bufferSize;
    public ComputeShader LaplaceCS_1;   // Phase1 odd
    public ComputeShader LaplaceCS_2;   // Phase2 even
    ComputeBuffer potential_buffer_read, potential_buffer_write, phase1_to_2;
    #endregion GPU

    public int width = 256;
    public int height = 256;

    public float left_strength;
    public float right_strength;
    public float up_strength;
    public float bottom_strength;

    Texture2D texture;
    float[] potential_read, potential_write;

    void Start() {
        texture = new Texture2D(width, height, TextureFormat.ARGB32, false);
        texture.filterMode = FilterMode.Point;
        
        bufferSize = width * height;
        potential_read = new float[bufferSize];
        potential_write = new float[bufferSize];
        threadGroupSize = Mathf.CeilToInt(bufferSize / SIMULATION_BLOCK_SIZE) + 1;
        potential_buffer_read = new ComputeBuffer(bufferSize, sizeof(float));
        potential_buffer_write = new ComputeBuffer(bufferSize, sizeof(float));
        phase1_to_2 = new ComputeBuffer(bufferSize, sizeof(float));
        
        SetBoundaryCondition();

        LaplaceEquation();
        
    }

    void Update() {

    }

    void OnGUI() {
        GUI.DrawTexture(new Rect(new Vector2(0, 0), new Vector2(texture.width, texture.height)), texture);
    }

    void OnDestroy() {
        potential_buffer_read.Release();
        potential_buffer_write.Release();
        phase1_to_2.Release();
    }

    void SetBoundaryCondition() {

        for (int i = 0; i < bufferSize; i++) {
            potential_read[i] = 0;
        }

        for (int i = 0; i < bufferSize; i++) {
            if (i < width) potential_read[i] = up_strength;
            if(i >= bufferSize - width) potential_read[i] = bottom_strength;
            if(i % width == 0) potential_read[i] = left_strength;
            if(i % width == width - 1) potential_read[i] = right_strength;
        }

        potential_buffer_read.SetData(potential_read);
        potential_buffer_write.SetData(potential_read);
        
    }

    // For Static Mode
    void LaplaceEquation() {

        if(mode == Mode.Both || mode == Mode.Phase1) {
            LaplaceCS_1.SetInt("BUFFER_SIZE", bufferSize);
            LaplaceCS_1.SetInt("WIDTH", width);
            LaplaceCS_1.SetInt("HEIGHT", height);
            int kernel = LaplaceCS_1.FindKernel("Laplace_Phase1");
            LaplaceCS_1.SetBuffer(kernel, "_PotentialBufferRead", potential_buffer_read);
            LaplaceCS_1.SetBuffer(kernel, "_PotentialBufferWrite", phase1_to_2);

            LaplaceCS_1.Dispatch(kernel, threadGroupSize, 1, 1);

            SwapBuffer(ref potential_buffer_read, ref phase1_to_2);
        }
        
        if(mode == Mode.Both || mode == Mode.Phase2) {
            LaplaceCS_2.SetInt("BUFFER_SIZE", bufferSize);
            LaplaceCS_2.SetInt("WIDTH", width);
            LaplaceCS_2.SetInt("HEIGHT", height);

            int kernel = LaplaceCS_2.FindKernel("Laplace_Phase2");
            LaplaceCS_2.SetBuffer(kernel, "_PotentialBufferRead", phase1_to_2);
            LaplaceCS_2.SetBuffer(kernel, "_PotentialBufferWrite", potential_buffer_write);

            LaplaceCS_2.Dispatch(kernel, threadGroupSize, 1, 1);

            SwapBuffer(ref potential_buffer_read, ref potential_buffer_write);
        }

        // Get Data
        potential_buffer_read.GetData(potential_read);
        potential_buffer_write.GetData(potential_write);
        
        ApplyTexture();
    }

    void SwapBuffer(ref ComputeBuffer src, ref ComputeBuffer dst) {
        ComputeBuffer tmp = dst;
        dst = src;
        src = tmp;
    }

    void ApplyTexture() {
        for (int i = 0; i < bufferSize; i++) {
            texture.SetPixel(i % width, (height-1) - i/width, Color.HSVToRGB(0.5f + potential_read[i] * 0.5f, 1f, 1f));
        }
        texture.Apply();
    }
}
