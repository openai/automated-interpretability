import { TokenAndActivation } from "./types"
import TokenHeatmap from "./tokenHeatmap";

export default ({ allTokens }: { allTokens: TokenAndActivation[][]}) => {
  return (
    <div className="">
      {allTokens.map((tokens, i) => (
        <div className="block my-3 border p-3 m-2 rounded-md" style={{ }} key={i}>
          <TokenHeatmap tokens={tokens} />
        </div>
      ))}
    </div>
  );
};
