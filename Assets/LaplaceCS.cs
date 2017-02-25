using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LaplaceCS : MonoBehaviour {

    public enum Mode {
        Animation, Static
    }
    public Mode mode;

    #region GPU
    const int SIMULATION_BLOCK_SIZE = 100;
    int threadGroupSize;
    int bufferSize;
    public ComputeShader LaplaceCS_1;   // Phase1 odd
    public ComputeShader LaplaceCS_2;   // Phase2 even
    ComputeBuffer potential_buffer_read, potential_buffer_write, phase1_to_2;
    #endregion GPU

    public int width = 256;
    public int height = 256;

    public float allowed_error = 0.00001f;
    public int allowed_iter = 1000;

    public float left_strength;
    public float right_strength;
    public float up_strength;
    public float bottom_strength;

    [Range(1.0f, 1.3f)] public float sor_coef = 1f;

    Texture2D texture;
    float[] potential_read, potential_write;
    
    int count;
    bool finish;
    float errorMax;

    void Start() {
        texture = new Texture2D(width, height, TextureFormat.ARGB32, false);
        texture.filterMode = FilterMode.Point;

        count = 0;
        finish = false;

        bufferSize = width * height;
        potential_read = new float[bufferSize];
        potential_write = new float[bufferSize];
        threadGroupSize = Mathf.CeilToInt(bufferSize / SIMULATION_BLOCK_SIZE) + 1;
        potential_buffer_read = new ComputeBuffer(bufferSize, sizeof(float));
        potential_buffer_write = new ComputeBuffer(bufferSize, sizeof(float));
        phase1_to_2 = new ComputeBuffer(bufferSize, sizeof(float));
        
        SetBoundaryCondition();

        if (!finish && mode == Mode.Static) LaplaceEquation();
    }

    void Update() {

        errorMax = 0f;
        if (!finish && mode == Mode.Animation) AnimatedLaplaceEquation();

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

        do {
            errorMax = 0;

            // Phase 1
            LaplaceCS_1.SetFloat("SOR_COEF", sor_coef);
            LaplaceCS_1.SetInt("BUFFER_SIZE", bufferSize);
            LaplaceCS_1.SetInt("WIDTH", width);
            LaplaceCS_1.SetInt("HEIGHT", height);

            int kernel = LaplaceCS_1.FindKernel("Laplace_Phase1");
            LaplaceCS_1.SetBuffer(kernel, "_PotentialBufferRead", potential_buffer_read);
            LaplaceCS_1.SetBuffer(kernel, "_PotentialBufferWrite", phase1_to_2);

            LaplaceCS_1.Dispatch(kernel, threadGroupSize, 1, 1);


            // Phase 2
            LaplaceCS_2.SetFloat("SOR_COEF", sor_coef);
            LaplaceCS_2.SetInt("BUFFER_SIZE", bufferSize);
            LaplaceCS_2.SetInt("WIDTH", width);
            LaplaceCS_2.SetInt("HEIGHT", height);

            kernel = LaplaceCS_2.FindKernel("Laplace_Phase2");
            LaplaceCS_2.SetBuffer(kernel, "_PotentialBufferRead", phase1_to_2);
            LaplaceCS_2.SetBuffer(kernel, "_PotentialBufferWrite", potential_buffer_write);

            LaplaceCS_2.Dispatch(kernel, threadGroupSize, 1, 1);


            // Error Check
            potential_buffer_read.GetData(potential_read);
            potential_buffer_write.GetData(potential_write);

            float error;
            for (int i = 0; i < bufferSize; i++) {
                error = Mathf.Abs(potential_read[i] - potential_write[i]);
                if (errorMax < error) {
                    errorMax = error;
                }
            }

            // Debug.Log(count++ + ", " + errorMax);

            // Buffer Swap
            SwapBuffer();

            count++;
            if (count > allowed_iter) break;

        } while (errorMax > allowed_error);

        finish = true;
        Debug.Log("Finished: " + count);

        ApplyTexture();
    }

    // For Animation Mode
    void AnimatedLaplaceEquation() {

        // Phase 1
        LaplaceCS_1.SetFloat("SOR_COEF", sor_coef);
        LaplaceCS_1.SetInt("BUFFER_SIZE", bufferSize);
        LaplaceCS_1.SetInt("WIDTH", width);
        LaplaceCS_1.SetInt("HEIGHT", height);

        int kernel = LaplaceCS_1.FindKernel("Laplace_Phase1");
        LaplaceCS_1.SetBuffer(kernel, "_PotentialBufferRead", potential_buffer_read);
        LaplaceCS_1.SetBuffer(kernel, "_PotentialBufferWrite", phase1_to_2);

        LaplaceCS_1.Dispatch(kernel, threadGroupSize, 1, 1);


        // Phase 2
        LaplaceCS_2.SetFloat("SOR_COEF", sor_coef);
        LaplaceCS_2.SetInt("BUFFER_SIZE", bufferSize);
        LaplaceCS_2.SetInt("WIDTH", width);
        LaplaceCS_2.SetInt("HEIGHT", height);

        kernel = LaplaceCS_2.FindKernel("Laplace_Phase2");
        LaplaceCS_2.SetBuffer(kernel, "_PotentialBufferRead", phase1_to_2);
        LaplaceCS_2.SetBuffer(kernel, "_PotentialBufferWrite", potential_buffer_write);

        LaplaceCS_2.Dispatch(kernel, threadGroupSize, 1, 1);


        // Error Check
        potential_buffer_read.GetData(potential_read);
        potential_buffer_write.GetData(potential_write);

        float error;
        for (int i = 0; i < bufferSize; i++) {
            error = Mathf.Abs(potential_read[i] - potential_write[i]);
            if (errorMax < error) {
                errorMax = error;
            }
        }

        Debug.Log(count++ + ", " + errorMax);

        if (errorMax < allowed_error) {
            finish = true;
        }


        // Result
        ApplyTexture();

        // Buffer Swap
        SwapBuffer();

    }

    void SwapBuffer() {
        ComputeBuffer tmp = potential_buffer_write;
        potential_buffer_write = potential_buffer_read;
        potential_buffer_read = tmp;
    }

    void ApplyTexture() {
        for (int i = 0; i < bufferSize; i++) {
            texture.SetPixel(i % width, (height-1) - i/width, new Color(0f, 0f, potential_read[i]));
        }
        texture.Apply();
    }
}
