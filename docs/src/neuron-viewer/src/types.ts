import { scaleLinear } from "d3-scale"
import { min, max, flatten } from "lodash"

export type Neuron = {
  layer: number;
  neuron: number;
}

export type TokenAndActivation = {
  token: string,
  activation: number
  normalized_activation?: number
}

export type TokenSequence = TokenAndActivation[]

export const normalizeTokenActs = (...sequences: TokenSequence[][]) => {
  // console.log('sequences', sequences)
  let flattened: TokenAndActivation[] = flatten(flatten(sequences))
  // Replace all activations less than 0 in data.tokens with 0. This matches the format in the
  // top + random activation records displayed in the main grid.
  flattened = flattened.map(({token, activation}) => {
    return {
      token,
      activation: Math.max(activation, 0)
    }
  })
  const maxActivation = max(flattened.map((ta) => ta.activation)) || 0;
  const neuronScale = scaleLinear()
    // Even though we're only displaying positive activations, we still need to scale in a way that
    // accounts for the existence of negative activations, since our color scale includes them.
    .domain([0, maxActivation])
    .range([0, 1])

  return sequences.map((seq) => seq.map((tas) => tas.map(({ token, activation }) => ({
      token,
      activation,
      normalized_activation: neuronScale(activation),
  }))))
}

export type Color = {r: number, g: number, b: number};
export function interpolateColor(color_l: Color, color_r: Color, value: number) {
  const color = {
    r: Math.round(color_l.r + (color_r.r - color_l.r) * value),
    g: Math.round(color_l.g + (color_r.g - color_l.g) * value),
    b: Math.round(color_l.b + (color_r.b - color_l.b) * value),
  }
  return color
}

export function getInterpolatedColor(colors: Color[], boundaries: number[], value: number) {
  const index = boundaries.findIndex((boundary) => boundary >= value)
  const colorIndex = Math.max(0, index - 1)
  const color_left = colors[colorIndex]
  const color_right = colors[colorIndex + 1]
  const boundary_left = boundaries[colorIndex]
  const boundary_right = boundaries[colorIndex + 1]
  const ratio = (value - boundary_left) / (boundary_right - boundary_left)
  const color = interpolateColor(color_left, color_right, ratio)
  return color
}

export const DEFAULT_COLORS = [
  // { r: 255, g: 0, b: 105 },
  { r: 255, g: 255, b: 255 },
  { r: 0, g: 255, b: 0 },
]
export const DEFAULT_BOUNDARIES = [
  // 0, 0.5, 1
  0, 1
]

