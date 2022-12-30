
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;
using System;
using System.Collections.Generic;
using Unity.Barracuda;

#if CJM_BARRACUDA_INFERENCE && CJM_BBOX_2D_TOOLKIT && CJM_YOLOX_UTILS
using CJM.YOLOXUtils;
using CJM.BBox2DToolkit;

namespace CJM.BarracudaInference.YOLOX
{
    /// <summary>
    /// YOLOXObjectDetector is a class that extends BarracudaModelRunner for object detection using the YOLOX model.
    /// It handles model execution, processes the output, and generates bounding boxes with corresponding labels and colors.
    /// The class supports various worker types, including PixelShader and ComputePrecompiled, as well as Async GPU Readback.
    /// </summary>
    public class YOLOXObjectDetector : BarracudaModelRunner
    {
        // Output Processing configuration and variables
        [Header("Output Processing")]
        // JSON file containing the color map for bounding boxes
        [SerializeField, Tooltip("JSON file with bounding box colormaps")]
        private TextAsset colormapFile;

        [Header("Settings")]
        [Tooltip("Interval (in frames) for unloading unused assets with Pixel Shader backend")]
        [SerializeField] private int pixelShaderUnloadInterval = 100;

        // A counter for the number of frames processed.
        private int frameCounter = 0;

        // Indicates if the system supports asynchronous GPU readback
        private bool supportsAsyncGPUReadback = false;

        // Stride values used by the YOLOX model
        private static readonly int[] Strides = { 8, 16, 32 };

        // Number of fields in each bounding box
        private const int NumBBoxFields = 5;

        // Layer names for the Transpose, Flatten, and TransposeOutput operations
        private const string TransposeLayer = "transpose";
        private const string FlattenLayer = "flatten";
        private const string TransposeOutputLayer = "transposeOutput";
        private string defaultOutputLayer;

        // Texture formats for output processing
        private TextureFormat textureFormat = TextureFormat.RHalf;
        private RenderTextureFormat renderTextureFormat = RenderTextureFormat.RHalf;

        // Serializable classes to store color map information from JSON
        [System.Serializable]
        class Colormap
        {
            public string label;
            public List<float> color;
        }

        [System.Serializable]
        class ColormapList
        {
            public List<Colormap> items;
        }

        // List to store label and color pairs for each class
        private List<(string, Color)> colormapList = new List<(string, Color)>();

        // Output textures for processing on CPU and GPU
        private Texture2D outputTextureCPU;
        private RenderTexture outputTextureGPU;

        // List to store grid and stride information for the YOLOX model
        private List<GridCoordinateAndStride> gridCoordsAndStrides = new List<GridCoordinateAndStride>();

        // Length of the proposal array for YOLOX output
        private int proposalLength;

        // Called at the start of the script
        protected override void Start()
        {
            base.Start();
            CheckAsyncGPUReadbackSupport(); // Check if async GPU readback is supported
            LoadColorMapList(); // Load colormap information from JSON file
            CreateOutputTexture(1, 1); // Initialize output texture

            proposalLength = colormapList.Count + NumBBoxFields; // Calculate proposal length
        }

        // Check if the system supports async GPU readback
        public bool CheckAsyncGPUReadbackSupport()
        {
            supportsAsyncGPUReadback = SystemInfo.supportsAsyncGPUReadback && supportsAsyncGPUReadback;
            return supportsAsyncGPUReadback;
        }

        // Load and prepare the YOLOX model
        protected override void LoadAndPrepareModel()
        {
            base.LoadAndPrepareModel();

            defaultOutputLayer = modelBuilder.model.outputs[0];
            WorkerFactory.Type bestType = WorkerFactory.ValidateType(WorkerFactory.Type.Auto);
            bool supportsComputeBackend = bestType == WorkerFactory.Type.ComputePrecompiled;

            // Set worker type for WebGL
            if (Application.platform == RuntimePlatform.WebGLPlayer)
            {
                workerType = WorkerFactory.Type.PixelShader;
            }

            // Apply transpose operation on the output layer
            modelBuilder.Transpose(TransposeLayer, defaultOutputLayer, new[] { 0, 3, 2, 1, });
            defaultOutputLayer = TransposeLayer;

            // Apply Flatten and TransposeOutput operations if supported
            if (supportsComputeBackend && (workerType != WorkerFactory.Type.PixelShader))
            {
                modelBuilder.Flatten(FlattenLayer, TransposeLayer);
                modelBuilder.Transpose(TransposeOutputLayer, FlattenLayer, new[] { 0, 1, 3, 2 });
                modelBuilder.Output(TransposeLayer);
                defaultOutputLayer = TransposeOutputLayer;
            }
        }

        /// <summary>
        /// Initialize the Barracuda engine
        /// <summary>
        protected override void InitializeEngine()
        {
            base.InitializeEngine();

            // Check if async GPU readback is supported by the engine
            supportsAsyncGPUReadback = engine.Summary().Contains("Unity.Barracuda.ComputeVarsWithSharedModel");
        }

        /// <summary>
        /// Load the color map list from the JSON file
        /// <summary>
        private void LoadColorMapList()
        {
            if (IsColorMapListJsonNullOrEmpty())
            {
                Debug.LogError("Class labels JSON is null or empty.");
                return;
            }

            ColormapList colormapObj = DeserializeColorMapList(colormapFile.text);
            UpdateColorMap(colormapObj);
        }

        /// <summary>
        /// Check if the color map JSON file is null or empty
        /// <summary>
        private bool IsColorMapListJsonNullOrEmpty()
        {
            return colormapFile == null || string.IsNullOrWhiteSpace(colormapFile.text);
        }

        /// <summary>
        /// Deserialize the color map list from the JSON string
        /// <summary>
        private ColormapList DeserializeColorMapList(string json)
        {
            try
            {
                return JsonUtility.FromJson<ColormapList>(json);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to deserialize class labels JSON: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Update the color map list with deserialized data
        /// <summary>
        private void UpdateColorMap(ColormapList colormapObj)
        {
            if (colormapObj == null)
            {
                return;
            }

            // Add label and color pairs to the colormap list
            foreach (Colormap colormap in colormapObj.items)
            {
                Color color = new Color(colormap.color[0], colormap.color[1], colormap.color[2]);
                colormapList.Add((colormap.label, color));
            }
        }

        /// <summary>
        /// Create an output texture with the specified width and height.
        /// </summary>
        private void CreateOutputTexture(int width, int height)
        {
            outputTextureCPU = new Texture2D(width, height, textureFormat, false);
        }

        /// <summary>
        /// Execute the YOLOX model with the given input texture.
        /// </summary>
        public void ExecuteModel(RenderTexture inputTexture)
        {
            using (Tensor input = new Tensor(inputTexture, channels: 3))
            {
                base.ExecuteModel(input);
            }

            // Update grid_strides if necessary
            if (engine.PeekOutput(defaultOutputLayer).length / proposalLength != gridCoordsAndStrides.Count)
            {
                gridCoordsAndStrides = YOLOXUtility.GenerateGridCoordinatesWithStrides(Strides, inputTexture.height, inputTexture.width);
            }
        }

        /// <summary>
        /// Process the output array from the YOLOX model, applying Non-Maximum Suppression (NMS) and
        /// returning an array of BBox2DInfo objects with class labels and colors.
        /// </summary>
        /// <param name="outputArray">The output array from the YOLOX model</param>
        /// <param name="confidenceThreshold">The minimum confidence score for a bounding box to be considered</param>
        /// <param name="nms_threshold">The threshold for Non-Maximum Suppression (NMS)</param>
        /// <returns>An array of BBox2DInfo objects containing the filtered bounding boxes, class labels, and colors</returns>
        public BBox2DInfo[] ProcessOutput(float[] outputArray, float confidenceThreshold = 0.5f, float nms_threshold = 0.45f)
        {
            // Generate bounding box proposals from the output array
            List<BBox2D> proposals = YOLOXUtility.GenerateBoundingBoxProposals(outputArray, gridCoordsAndStrides, colormapList.Count, NumBBoxFields, confidenceThreshold);

            // Apply Non-Maximum Suppression (NMS) to the proposals
            List<int> proposal_indices = BBox2DUtility.NMSSortedBoxes(proposals, nms_threshold);

            // Create an array of BBox2DInfo objects containing the filtered bounding boxes, class labels, and colors