import React, { useRef, useState, useEffect } from "react";
import axios from "axios";  // Ensure axios is imported for making HTTP requests

const CameraApp = () => {
  const videoRef = useRef(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [detectionResults, setDetectionResults] = useState(null);

  // Start the camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraOn(true);
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert("Unable to access the camera. Please check your permissions.");
    }
  };

  // Stop the camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      let tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraOn(false);
  };

  // Capture video frame and send to FastAPI for mask detection
  const captureAndDetectMask = async () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      // Draw the current video frame onto the canvas
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      // Convert canvas image to base64 format
      const imageUrl = canvas.toDataURL("image/jpeg");
    //   console.log(imageUrl)

      try {
        // Send the base64 image to the FastAPI backend for mask detection
        const response = await axios.post("http://localhost:8000/detect_mask", {
          image: imageUrl, // Send the image as a base64 string
        });

        console.log(response)

        // Store the results (e.g., mask detection predictions)
        setDetectionResults(response.data.results);
      } catch (error) {
        console.error("Error sending frame to backend:", error);
      }
    }
  };

  // Periodically capture and send frames to backend
  useEffect(() => {
    const interval = setInterval(() => {
      if (isCameraOn) {
        captureAndDetectMask();
      }
    }, 1000); // Capture frame every 1 second

    return () => clearInterval(interval); // Cleanup interval on unmount
  }, [isCameraOn]);

  return (
    <div>
      {/* Navigation Bar */}
      <nav className="flex justify-between items-center p-4 bg-blue-500 text-white sticky top-0 z-10">
        <h1 className="m-0 text-xl">Mask Guard</h1>
        <div>
          <button className="px-4 py-2 text-lg cursor-pointer mx-2 bg-white text-blue-500 rounded">Login</button>
          <button className="px-4 py-2 text-lg cursor-pointer mx-2 bg-white text-blue-500 rounded">Sign Up</button>
        </div>
      </nav>

      {/* Main Content */}
      <div className="text-center mt-6">
        <video ref={videoRef} autoPlay className="transform scale-x-[-1]"></video> {/* Mirroring the video */}
        <div className="mt-4">
          {!isCameraOn ? (
            <button onClick={startCamera} className="px-6 py-3 text-lg cursor-pointer bg-blue-500 text-white rounded">
              Open Camera
            </button>
          ) : (
            <button onClick={stopCamera} className="px-6 py-3 text-lg cursor-pointer bg-blue-500 text-white rounded">
              Stop Camera
            </button>
          )}
        </div>

        {/* Display Detection Results */}
        {detectionResults && (
          <div className="mt-6">
            <h3 className="text-xl">Detection Results:</h3>
            {detectionResults.map((result, index) => (
              <div key={index} className="mt-4">
                <p>
                  <strong>Label:</strong> {result.label}
                </p>
                <p>
                  <strong>Confidence:</strong> {result.confidence * 100}%
                </p>
                <p>
                  <strong>Bounding Box:</strong> {result.box.join(", ")}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraApp;
