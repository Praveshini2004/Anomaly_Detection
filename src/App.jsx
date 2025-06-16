import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom"; 
import ImageGenerator from "./ImageGenerator";

function App() {
  return (
    <Router>
      <Routes> 
        <Route path="/" element={<ImageGenerator />} /> 
      </Routes>
    </Router>
  );
}

export default App;

