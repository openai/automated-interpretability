import React from "react"
import { interpolateColor, Color, getInterpolatedColor, DEFAULT_COLORS, DEFAULT_BOUNDARIES, TokenAndActivation } from './types'


type Props = {
  tokens: TokenAndActivation[], 
  loading?: boolean, 
  colors?: Color[], 
  boundaries?: number[]
}
export default function TokenHeatmap({ tokens, loading, colors = DEFAULT_COLORS, boundaries = DEFAULT_BOUNDARIES }: Props) {
    // <div className="block" style={{width:'100%', whiteSpace: 'pre', overflowX: 'scroll' }}>
  return (
    <div className="block" style={{width:'100%', whiteSpace: 'pre-wrap'}}>
      {tokens.map(({ token, activation, normalized_activation }, i) => {
        const color = getInterpolatedColor(colors, boundaries, normalized_activation || activation);
        return <span key={i}
          title={loading ? '' : `Activation: ${activation.toFixed(2)}`}
          className={`${loading ? "animate-pulse" : ""}`}
          style={{
            transition: "500ms ease-in all",
            background: loading
              ? `rgba(0, 0, 0, 0.03)`
              : `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`,
          }}
        >
          {token}
        </span>
      })}
    </div>
  )
}
