
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
    public class PackageList
    {
        public List<PackageData> packages;
    }

    public class PackageInstaller
    {
        // Stores the AddRequest object for the current package to install.
        private static AddRequest addRequest;
        // A list of PackageData objects to install.
        private static List<PackageData> packagesToInstall;
        // The index of the current package to install.
        private static int currentPackageIndex;

        // GUID of the JSON file containing the list of packages to install
        private const string PackagesJSONGUID = "02aec9cd479b4b758a7afde0032230ec";

        // Method called on load to install packages from the JSON file
        [InitializeOnLoadMethod]
        public static void InstallDependencies()
        {