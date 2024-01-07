import {Neuron} from './types';
import {memoizeAsync} from "./utils"

export const load_file_no_cache = async(path: string) => {
  const data = {
    path: path
  }
  const url = new URL("/load_az", window.location.href)
  url.port = '8000';
  return await (
    await fetch(url, {
      method: "POST", // or 'PUT'
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
  ).json()
  
}

export  const load_file_az = async(path: string) => {
  const res = (
    await fetch(path, {
      method: "GET",
      mode: "cors",
      headers: {
        "Content-Type": "application/json",
      },
    })
  )
  if (!res.ok) {
    console.error(`HTTP error: ${res.status} - ${res.statusText}`);
    return;
  }
  return await res.json()
}


// export const load_file = memoizeAsync('load_file', load_file_no_cache)
export  const load_file = window.location.host.indexOf('localhost:') === -1 ? load_file_az : load_file_no_cache;


// # (derived from az://oaialignment/datasets/interp/gpt2_xl/v1/webtext1/len_nomax/n_50000/mlp_post_act/ranked_by_max_activation)
// const NEURON_RECORDS_PATH = "az://oaisbills/rcall/oss/migrated_make_crow_datasets/gpt2_xl_n_50000_64_token/neurons"
const NEURON_RECORDS_PATH = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/collated-activations"

// # (derived from az://oaialignment/datasets/interp/gpt2_xl/v1/webtext1/len_nomax/n_50000/mlp_post_act/ranked_by_max_activation/neurons/explanations/canonical-run-v1)
// const EXPLANATIONS_PATH = "az://oaisbills/rcall/oss/migrated_explanation_datasets/canonical_gpt2_xl_all_neurons"
const EXPLANATIONS_PATH = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/explanations"

// weight-based
// const WHOLE_LAYER_WEIGHT_TOKENS_PATH = "az://oaidan/rcall/data/interpretability/connections/gpt2-xl/mlp/unnorm_token_representations_uncommon_vanilla"
// const WEIGHT_TOKENS_PATH = "az://oaijeffwu/jeffwu-data/interpretability/neuron-connections/gpt2-xl/weight-based"
const WEIGHT_TOKENS_PATH = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/related-tokens/weight-based"
// lookup table
// const WHOLE_LAYER_ACTIVATION_TOKENS_PATH = "az://oaidan/rcall/data/interpretability/connections/gpt2_xl/mlp/unnorm_token_representations_vanilla_and_common_in_colangv2_unigram"
// const ACTIVATION_TOKENS_PATH = "az://oaijeffwu/jeffwu-data/interpretability/neuron-connections/gpt2-xl/lookup-table"
const ACTIVATION_TOKENS_PATH = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/related-tokens/activation-based"

// const CONNECTIONS_PATH = "az://oaialignment/datasets/interp/connections/gpt2/neuron_space/incl_attn_False"
const CONNECTIONS_PATH = "https://openaipublic.blob.core.windows.net/neuron-explainer/data/related-neurons/weight-based"


export const get_explanations = async (activeNeuron: Neuron) => {
  const result = await load_file(`${EXPLANATIONS_PATH}/${activeNeuron.layer}/${activeNeuron.neuron}.jsonl`)
  return result
}

export const get_top_tokens = async (activeNeuron: Neuron, weightType: string) => {
  let TOKENS_PATH;
  if (weightType === 'weight') {
    TOKENS_PATH = WEIGHT_TOKENS_PATH;
  } else if (weightType === 'activation') {
    TOKENS_PATH = ACTIVATION_TOKENS_PATH;
  } else {
    throw new Error(`Invalid weightType: ${weightType}`)
  }
  const result = await load_file(`${TOKENS_PATH}/${activeNeuron.layer}/${activeNeuron.neuron}.json`)
  return result
  // const result = await load_file_no_cache(`${ORIG_TOKENS_PATH}/${activeNeuron.layer}.json`)
  // return result.neuron_summaries[activeNeuron.neuron]
}

export const get_top_neuron_connections = async (activeNeuron: Neuron) => {
    const result = await load_file(`${CONNECTIONS_PATH}/${activeNeuron.layer}/${activeNeuron.neuron}.json`)

    const res: {[key: string]: [number, number]} = {};
    ["input", "output"].forEach((direction) => {
        const sign = "positive"  // "negative"
        const weight_name: string = {output: "c_proj", input: "c_fc"}[direction] as string;
        const res_for_dir = result[weight_name];
        if (res_for_dir === null) {
          return
        }
        // let key = 'top_negative_neurons'
        const top_neuron_strs = res_for_dir[`top_${sign}_neurons`]  // {layer}_{neuron} strings for each top-connected neuron
        const top_weights = res_for_dir[`top_${sign}_weights`]
        const top_layer_neuron_tuples = top_neuron_strs.map((neuron_str: string, i: number) => {
            const [layer, neuron] = neuron_str.split("_").map((x: string) => parseInt(x))
            return [layer, neuron, top_weights[i]] as [number, number, number]
        })
        res[direction] = top_layer_neuron_tuples.slice(0, 10)
    })

    return res
}

export const get_neuron_record = async(activeNeuron: Neuron) => {
  const result = await load_file(`${NEURON_RECORDS_PATH}/${activeNeuron.layer}/${activeNeuron.neuron}.json`)
  return result
}

