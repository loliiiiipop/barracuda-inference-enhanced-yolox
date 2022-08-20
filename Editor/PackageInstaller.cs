
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CJM.BarracudaInference.YOLOX
{
    // Serializable class to hold package data
    [System.Serializable]
    public class PackageData
    {
        public string packageName;
        public string packageUrl;
    }

    // Serializable class to hold a list of PackageData objects
    [System.Serializable]