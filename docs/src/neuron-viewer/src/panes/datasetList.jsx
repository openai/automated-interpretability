import HeatmapGrid from "../heatmapGrid"
import React, { useEffect, useState } from "react"
import { normalizeTokenActs } from "../types"

import {get_neuron_record} from "../interpAPI"

function zip_sequences(sequences) {
  return sequences.map(({ activations, tokens }) => {
    return tokens.map((token, idx) => ({
      token,
      activation: activations[idx],
    }))
  })
}

export default ({ activeNeuron }) => {
  const [data, setData] = useState(null)
  const [showingMore, setShowingMore] = useState({})
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      if (data) {
        return
      }
      const result = await get_neuron_record(activeNeuron)
    console.log(result)
      const all_sequences = []
      all_sequences.push({
        // label: '[0.999, 1] (Top quantile, sorted.  50 of 50000)',
        label: 'Top',
        sequences: zip_sequences(result.most_positive_activation_records),
        default_show: 4,
      })
      all_sequences.push({
        label: 'Quantile range [0.99, 0.999] sample',
        sequences: zip_sequences(result.random_sample_by_quantile[3]),
        default_show: 1,
      })
      all_sequences.push({
        label: 'Quantile range [0.9, 0.99] sample',
        sequences: zip_sequences(result.random_sample_by_quantile[2]),
        default_show: 1,
      })
      all_sequences.push({
        label: 'Quantile range [0.5, 0.9] sample',
        sequences: zip_sequences(result.random_sample_by_quantile[1]),
        default_show: 1,
      })
      all_sequences.push({
        label: 'Quantile range [0, 0.5] sample',
        sequences: zip_sequences(result.random_sample_by_quantile[0]),
        default_show: 1,
      })
      all_sequences.push({
        // label: '[0, 1] (Random)',
        label: 'Random sample',
        sequences: zip_sequences(result.random_sample),
        default_show: 2,
      })
      // for reference
      // intervals = [(0, 1), (0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1)]
      // saved_activations_by_interval = [neuron_record.random_sample] + neuron_record.random_sample_by_decile[:-1] + [neuron_record.top_activations]
      setData(all_sequences)
      setIsLoading(false)
    }
    fetchData()
  }, [activeNeuron])

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="w-8 h-8 border-4 border-gray-300 rounded-full animate-spin"></div>
        <div>loading top dataset examples</div>
      </div>
    )
  }

  // const activations = data.top_activations;
  const all_normalized_sequences = normalizeTokenActs(...data.map(({sequences}) => sequences))

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Activations</h2>
      {
        data.map(({label, default_show}, idx) => {
          const n_show = showingMore[label] ? all_normalized_sequences[idx].length : default_show;
          return (
          <React.Fragment key={idx}>
          <h3 className="text-md font-bold">
            {label}
            <button className="ml-2 text-sm text-gray-500"
              onClick={() => setShowingMore({...showingMore, [label]: !showingMore[label]})}>
              {showingMore[label] ? 'show less' : 'show more'}
            </button>
          </h3>
          <HeatmapGrid allTokens={all_normalized_sequences[idx].slice(0, n_show)} />
          </React.Fragment>
          )
        })
      }
    </div>
  )
}
