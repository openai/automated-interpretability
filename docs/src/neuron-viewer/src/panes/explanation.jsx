import React, { useState, useEffect } from "react"
import { get_explanations } from "../interpAPI"
// import HeatmapGrid from "../heatmapGrid"
import SimulationHeatmap from "../simulationHeatmap"
import { normalizeTokenActs } from "../types"


function zip_simulated_sequences(sequences) {
  return sequences.map(({ simulation }) => {
    return simulation.tokens.map((token, idx) => ({
      token,
      activation: simulation.expected_activations[idx],
    }))
  })
}

function zip_real_sequences(sequences) {
  return sequences.map(({ simulation, true_activations }) => {
    return simulation.tokens.map((token, idx) => ({
      token,
      activation: true_activations[idx],
    }))
  })
}

const ExplanationDisplay = ({ activeNeuron }) => {
  const [isLoading, setIsLoading] = useState(true)
  const [data, setData] = useState(null)
  const [showingScoringDetails, setShowingScoringDetails] = useState(false)
  const [toggle, setToggle] = useState(false);

  const loadExplanation = async () => {
    const result = await get_explanations(activeNeuron);
    setData(result.scored_explanations[0])
    setIsLoading(false)
  }

  useEffect(() => {
    if (!data) {
      loadExplanation()
    }
  }, [])

  const handleToggleChange = () => {
    setToggle(!toggle);
  };

  let sim_sequences;
  if (data) {
    sim_sequences = zip_simulated_sequences(data.scored_simulation.scored_sequence_simulations);
    [sim_sequences] = normalizeTokenActs(sim_sequences)
  } else {
    sim_sequences = []
  }

  let real_sequences;
  if (data) {
    real_sequences = zip_real_sequences(data.scored_simulation.scored_sequence_simulations);
    [real_sequences] = normalizeTokenActs(real_sequences)
  } else {
    real_sequences = []
  }

  const suggest_explanation_link = "https://docs.google.com/forms/d/e/1FAIpQLSckMyDQedGhdISIqaqn0YGUtd2xqEWgPu7ehoPUTT2pTge_-g/viewform?"
    + `usp=pp_url&entry.541490611=${activeNeuron.layer}`
    + `&entry.1688855196=${activeNeuron.neuron}`
    + `&entry.495312202=https://openaipublic.blob.core.windows.net/neuron-explainer/neuron-viewer/index.html%23/layers/${activeNeuron.layer}/neurons/${activeNeuron.neuron}`;

  return (
    <>
      <div className="min-w-0 flex-1">
        <h2 className="text-2xl font-bold mb-4">Explanation</h2>
        {isLoading ? (
          <div className="flex justify-center items-center">
            <div className="loader">Loading...</div>
          </div>
        ) : (
          <>
            <blockquote className="p-1 px-4 mx-1 my-0">
              <p className="py-1">
                <em>{data.explanation}</em>
              </p>
              <p className="py-1">
                <em>score: {data.scored_simulation.ev_correlation_score.toFixed(2)}</em>
              </p>
              <p className="py-1">
                <a href={suggest_explanation_link}>Suggest Better Explanation</a>
              </p>
            </blockquote>
            <button onClick={() => { setShowingScoringDetails(!showingScoringDetails) }}>
              {showingScoringDetails ? 'Hide' : 'Show'} scoring details
            </button>
            {
              showingScoringDetails ?
                <>
                  <div
                    style={{
                      textAlign: 'right',
                    }}
                  >
                    <div
                      style={{
                        display: 'inline-block',
                        position: 'relative',
                        width: '60px',
                        height: '34px',
                        marginLeft: '10px',
                        marginBottom: '5px',
                        borderRadius: '34px',
                        backgroundColor: toggle ? '#0A978B' : '#CCC',
                        cursor: 'pointer',
                        transition: 'background-color 0.2s',
                      }}
                    >
                      <input
                        type="checkbox"
                        id="toggle"
                        checked={toggle}
                        onChange={handleToggleChange}
                        style={{
                          width: '100%',
                          height: '100%',
                          margin: '0',
                          opacity: '0',
                          cursor: 'pointer',
                        }}
                      />
                      <span
                        onClick={handleToggleChange}
                        style={{
                          position: 'absolute',
                          top: '2px',
                          left: toggle ? '29px' : '2px',
                          width: '30px',
                          height: '30px',
                          backgroundColor: 'white',
                          borderRadius: '50%',
                          boxShadow: '0 2px 5px rgba(0, 0, 0, 0.3)',
                          transition: 'left 0.2s',
                        }}
                      ></span>
                    </div>
                    <br />
                    {toggle ? 'Activations overlaid (top = real, bottom = simulated)' : 'Activations not overlaid'}
                  </div>
                  <h3 className="text-md font-bold">Top</h3>
                  <SimulationHeatmap
                    sequences={real_sequences.slice(0, 5)}
                    simulated_sequences={sim_sequences.slice(0, 5)}
                    overlay_activations={toggle}
                  />
                  <h3 className="text-md font-bold">Random</h3>
                  <SimulationHeatmap
                    sequences={real_sequences.slice(5)}
                    simulated_sequences={sim_sequences.slice(5)}
                    overlay_activations={toggle}
                  />
                </> : null
            }
          </>
        )}
      </div>
    </>
  )
}

export default ExplanationDisplay
