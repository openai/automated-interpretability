import "./App.css"
import Feed from "./feed"
import React from "react"
import { Routes, Route, HashRouter } from "react-router-dom"

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<Feed />} />
        <Route path="/layers/:layer/neurons/:neuron" element={<Feed />} />
      </Routes>
    </HashRouter>
  )
}

export default App
