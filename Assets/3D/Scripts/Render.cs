using UnityEngine;

namespace Laplace3D {

    [RequireComponent(typeof(Laplace))]
    public class Render : MonoBehaviour {

        public Laplace GPUScript;

        public Material ParticleRenderMat;

        void OnRenderObject() {
            DrawObject();
        }

        void DrawObject() {
            Material m = ParticleRenderMat;
            m.SetPass(0);
            m.SetBuffer("_Cells", GPUScript.GetBuffer());
            Graphics.DrawProcedural(MeshTopology.Points, GPUScript.GetBufferSize());
        }

    }

}