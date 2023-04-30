import * as Panes from "./panes"
import React, { useEffect } from "react"
import Welcome from "./welcome"
import { useState } from "react"
import { useParams, Link } from "react-router-dom"

export default function Feed() {
  const params = useParams()
  // If params is missing either index, there's no neuron selected.
  let activeNeuron;
  if (params.layer === undefined || params.neuron === undefined) {
    activeNeuron = null
  } else {
    // Grab the layer and neuron indices from the params, casting them to ints.
    activeNeuron = {
      "layer": parseInt(params.layer),
      "neuron": parseInt(params.neuron),
    }
  }

  const Pane = ({ children }) => (
    <div className="flex flex-col h-full">{children}</div>
  )

  return (
    <div>
      <div>
        <h2 className="flex flex-row">
          <Link to="/">Neuron Viewer</Link>
        </h2>
        {activeNeuron && (
          <h3 className="flex flex-row">
            Neuron {activeNeuron.layer}:{activeNeuron.neuron}
          </h3>
        )}
      </div>

      <div
        style={{ width: '100%', padding: '0px 80px', margin: "auto", overflow: "visible" }}
      >
        <ul role="list" className="mb-8 mt-10">
          {activeNeuron ?
            <>
              <Pane>
                {React.createElement(Panes["Explanation"], { activeNeuron })}
              </Pane>
              <Pane>
                {React.createElement(Panes["DatasetList"], { activeNeuron })}
              </Pane>
              <Pane>
                {React.createElement(Panes["TopTokens"], { activeNeuron })}
              </Pane>
              <Pane>
                {React.createElement(Panes["SimilarNeurons"], { activeNeuron })}
              </Pane>
            </> :
            <Welcome/>
          }

        </ul>
      </div>
    </div>
  )
}
