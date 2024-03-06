import React, { useEffect, useState } from "react"
import _ from "lodash"
import { Link } from "react-router-dom"

import { get_explanations, get_top_neuron_connections } from "../interpAPI"

function NeuronInfo({ neuron, strength }) {
  const [info, setInfo] = useState(null)

  useEffect(() => {
    async function fetchInfo() {
      const result = (await get_explanations({
        layer: neuron.layer,
        neuron: neuron.neuron,
      }))
      setInfo(result)
    }

    if (!info) {
      fetchInfo()
    }
  }, [])

  if (!info) {
    return (
      <div className="m-4 flex justify-center items-center h-32">
        <p className="text-gray-500 mb-2">
          Loading neuron {neuron.layer}:{neuron.neuron}...
        </p>
        <div className="w-8 h-8 border-4 border-gray-300 rounded-full animate-spin"></div>
      </div>
    )
  }

  return (
    <div>
      <div className="overflow-hidden mb-4 border rounded-lg bg-white shadow">
        <h3
          className="px-4 text-lg pb-0 mb-0 font-bold">
          <Link to={`/layers/${neuron.layer}/neurons/${neuron.neuron}`}>
          Neuron {neuron.layer}:{neuron.neuron}
          </Link>
        </h3>
        <div className="text-sm px-4 py-2">
        Connection strength: {strength.toFixed(2)}
        </div>
        <blockquote className="p-1 px-4 mx-1 my-0">
          {info.scored_explanations.map((explanation, i) => (
            <React.Fragment key={i}>
            <p className="py-1">
              <em>{explanation.explanation}</em>
            </p>
            <p className="py-1">
              <em>score: {explanation.scored_simulation.ev_correlation_score.toFixed(2)}</em>
            </p>
            </React.Fragment>
          ))}
        </blockquote>
      </div>
    </div>
  )
}

export default function SimilarNeurons({ activeNeuron: neuron }) {
  const [similarNeurons, setSimilarNeurons] = useState([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    async function fetchSimilarNeurons() {
      const result = await get_top_neuron_connections(neuron)
      setSimilarNeurons(result)
      setIsLoading(false)
    }

    fetchSimilarNeurons()
  }, [neuron])

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="w-8 h-8 border-4 border-gray-300 rounded-full animate-spin"></div>
      </div>
    )
  }

  const n_show = 3;
  return (
    <div className="min-w-0 flex-1">
      <h2 className="text-2xl font-bold mb-4">Related neurons</h2>
      <div className="full-width mt-6">
        <div className="flex flow-row justify-center align-self-center">
          {
          similarNeurons.input ?
          <div style={{ width: 450 }} className="mx-2">
            <h5>Upstream</h5>
            <div className="flex flex-row flex-wrap">
              {similarNeurons.input.slice(0, n_show).map(([layer, neuron, strength]) => (
                <NeuronInfo key={layer + neuron} neuron={{ layer, neuron }} strength={strength} />
              ))}
            </div>
          </div> : null
          }
          {
          similarNeurons.output ?
          <div style={{ width: 450 }} className="mx-2">
            <h5>Downstream</h5>
            <div className="flex flex-row flex-wrap">
              {similarNeurons.output.slice(0, n_show).map(([layer, neuron, strength]) => (
                <NeuronInfo key={layer + neuron} neuron={{ layer, neuron }} strength={strength}/>
              ))}
            </div>
          </div> : null
          }
        </div>
      </div>
    </div>
  )
}
