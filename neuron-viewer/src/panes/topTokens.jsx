import React, { useState, useEffect } from "react"
import { get_top_tokens } from "../interpAPI"


const TokenDisplay = ({ activeNeuron }) => {
  const [isLoading, setIsLoading] = useState(true)
  const [data, setData] = useState(null)

  const loadTokens = async () => {
    setIsLoading(true)
    const weightStrengths = await get_top_tokens(activeNeuron, 'weight')
    const activationStrengths = await get_top_tokens(activeNeuron, 'activation')

    const data = {
      activeNeuron,
      weightStrengths,
      activationStrengths,
    }

    setData(data)
    setIsLoading(false)
  }

  useEffect(() => {
    if (!data) {
      loadTokens()
    }
  }, [])


  return (
    <div className="min-w-0 flex-1">
      <h2 className="text-2xl font-bold mb-4">Related tokens</h2>
      {isLoading ? (
        <div className="flex justify-center items-center">
          <div className="loader">loading tokens</div>
        </div>
      ) : (
        <>
          <h3 className="text-md font-bold mb-4">Mean-activation-based</h3>
          <div className="mt-2 text-sm text-gray-700">
            {data.activationStrengths.tokens.map((token, idx) => {
              return (
                data.activationStrengths.average_activations[idx] === null ? null :
                <span
                  key={idx}
                  title={`Strength: ${data.activationStrengths.average_activations[idx].toFixed(2)}`}
                  className="inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                >
                  {token}
                </span>
              )
            })}
          </div>
          <h3 className="text-md font-bold mb-4">Weight-based</h3>
          <div className="mt-2 text-sm text-gray-700">
            <p>Input tokens:</p>
            {data.weightStrengths.input_positive.tokens.slice(0, 20).map((token, idx) => {
              return (
                data.weightStrengths.input_positive.strengths[idx] === null ? null :
                <span
                  key={idx}
                  title={`Strength: ${data.weightStrengths.input_positive.strengths[idx].toFixed(2)}`}
                  className="inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                >
                  {token}
                </span>
              )
            })}
          </div>
          {
          <div className="mt-2 text-sm text-gray-700">
            <p>Input tokens negative:</p>
            {data.weightStrengths.input_negative.tokens.slice(0, 20).map((token, idx) => {
              return (
                data.weightStrengths.input_negative.strengths[idx] === null ? null :
                <span
                  key={idx}
                  title={`Strength: ${data.weightStrengths.input_negative.strengths[idx].toFixed(2)}`}
                  className="inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 text-red-800"
                >
                  {token}
                </span>
              )
            })}
          </div>
          }
          <div className="mt-2 text-sm text-gray-700">
            <p>Output tokens:</p>
            {data.weightStrengths.output_positive.tokens.slice(0, 20).map((token, idx) => {
              return (
                data.weightStrengths.output_positive.strengths[idx] === null ? null :
                <span
                  key={idx}
                  title={`Strength: ${data.weightStrengths.output_positive.strengths[idx].toFixed(2)}`}
                  className="inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                >
                  {token}
                </span>
              )
            })}
          </div>
          {
          <div className="mt-2 text-sm text-gray-700">
            <p>Output tokens negative:</p>
            {data.weightStrengths.output_negative.tokens.slice(0, 20).map((token, idx) => {
              return (
                <span
                  key={idx}
                  title={`Strength: ${data.weightStrengths.output_negative.strengths[idx].toFixed(2)}`}
                  className="inline-flex m-1 items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 text-red-800"
                >
                  {token}
                </span>
              )
            })}
          </div>
          }
        </>
      )}
    </div>
  )
}
export default TokenDisplay
