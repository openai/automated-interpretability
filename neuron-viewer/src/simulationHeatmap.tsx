import React, { useState } from 'react';

import { interpolateColor, Color, getInterpolatedColor, DEFAULT_COLORS, DEFAULT_BOUNDARIES, TokenAndActivation } from './types'

type Props = {
  sequences: TokenAndActivation[][], 
  simulated_sequences: TokenAndActivation[][], 
  overlay_activations: boolean,
  colors?: Color[], 
  boundaries?: number[],
}
export default function SimulationSequences({ sequences, simulated_sequences, overlay_activations, colors = DEFAULT_COLORS, boundaries = DEFAULT_BOUNDARIES }: Props) {
  return <>
    {
      sequences.map((tokens, i) => {
        let simulated_tokens = simulated_sequences[i];
        if (overlay_activations) {
          return (
            <div className="block my-3 border p-3 m-2 rounded-md" style={{ width: '100%' /*,whiteSpace: 'nowrap', overflowX: 'auto' */ }} key={i}>
              {tokens.map(({ token, activation, normalized_activation }, j) => {
                const { token: simulated_token, activation: simulated_activation, normalized_activation: simulated_normalized_activation } = simulated_tokens[j];
                if (simulated_token !== token) {
                  throw new Error('simulated tokens not matching')
                }
                const color = getInterpolatedColor(colors, boundaries, normalized_activation || activation);
                const simcolor = getInterpolatedColor(colors, boundaries, simulated_normalized_activation || simulated_activation);

                return <div style={{ display: 'inline-block', whiteSpace: 'pre' }} key={j}>
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <span
                      title={`Activation: ${activation.toFixed(2)}`}
                      style={{
                        transition: "500ms ease-in all",
                        background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`,
                      }}
                    >{token}</span>
                    <span
                      title={`Simulation: ${simulated_activation.toFixed(2)}`}
                      style={{
                        transition: "500ms ease-in all",
                        background: `rgba(${simcolor.r}, ${simcolor.g}, ${simcolor.b}, 0.5)`,
                      }}
                    >{token}</span>
                  </div>
                </div>
              })}
            </div>
          )
        } else {
          return (
            <div className="block my-3 border p-3 m-2 rounded-md" style={{ width: '100%' /*,whiteSpace: 'nowrap', overflowX: 'auto' */ }} key={i}>
              <div>
                <span
                  style={{
                    fontSize: '0.7em',
                    fontWeight: 'bold',
                  }}
                >Real activations:</span><br />
                {tokens.map(({ token, activation, normalized_activation }, j) => {
                  const color = getInterpolatedColor(colors, boundaries, normalized_activation || activation);
                  return <span key={j}
                    title={`Activation: ${activation.toFixed(2)}`}
                    style={{
                      transition: "500ms ease-in all",
                      background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`,
                    }}
                  >{token}</span>
                })}
              </div>
              <hr style={{ margin: '5px 0 5px 0' }} />
              <div>
                <span
                  style={{
                    fontSize: '0.7em',
                    fontWeight: 'bold',
                  }}
                >Simulated activations:</span><br />
                {simulated_tokens.map(({ token, activation, normalized_activation }, j) => {
                  const color = getInterpolatedColor(colors, boundaries, normalized_activation || activation);
                  return <span key={j}
                    title={`Activation: ${activation.toFixed(2)}`}
                    style={{
                      transition: "500ms ease-in all",
                      background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`,
                    }}
                  >{token}</span>
                })}
              </div>
            </div>
          )
        }
      })
    }
    </>
}
