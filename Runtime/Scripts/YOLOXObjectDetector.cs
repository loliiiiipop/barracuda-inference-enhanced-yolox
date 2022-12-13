
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