import { useState, FormEvent } from "react"
import { useNavigate } from "react-router-dom"

function NeuronForm() {
  const [input_layer, setLayer] = useState(0)
  const [input_neuron, setNeuron] = useState(0)
  const navigate = useNavigate()

  const knownGoodNeurons = [
    /**************
    /* well explained + interesting
    ***************/
    {heading: 'Somewhat well explained by GPT-4', layer: 0, neuron: 0, label: ''},
    {layer: 5, neuron: 131, label: "citations", description: "citations, especially biblical and legal"},
    {layer: 12, neuron: 847, label: "numbers in fractions", description: "numbers in fractions"}, // 
    {layer: 12, neuron: 5820, label: "short flags", description: "single letter command line flags"}, // 
    {layer: 14, neuron: 417, label: "doing things right", description: "words and phrases related to performing actions correctly or properly"}, // score 0.42
    {layer: 15, neuron: 4538, label: "leading transitions", description: "transition words at the start of documents"},
    {layer: 17, neuron: 3218, label: "success", description: "expressions of completion or success"}, // score 0.38
    {layer: 18, neuron: 5302, label: "X *by*", description: "the word 'by' in phrases indicating side by side or sequential events."}, // score 0.48
    {layer: 19, neuron: 1377, label: "similes", description: "comparisons and analogies, often using the word 'like'"}, // score 0.42
    {layer: 21, neuron: 2932, label: "Canada", description: "references to Canadian people, places, and entities"}, // score 0.78
    {layer: 25, neuron: 2602, label: "similes", description: "descriptive comparisons, especially similes"}, // score 0.40
    {layer: 25, neuron: 4870, label: "certainty", description: "phrases related to certainty and confidence."}, // score 0.37
    {layer: 30, neuron: 28, label: "times", description: "specific times (with hours and minutes)"}, 
    // https://openaipublic.blob.core.windows.net/neuron-explainer/neuron-viewer/index.html#/layers/5/neurons/2326
    {heading: 'Partially explained by GPT-4', layer: 0, neuron: 0, label: ''},
    {layer: 0, neuron: 816, label: "Marvel comics vibes", description: "language and context related to Marvel comics, movies, and characters, as well as other superhero-themed content"}, // score 0.44
    {layer: 0, neuron: 742, label: "Second token 'and'", description: "'and', 'in', and punctuation at the second token"},
    {layer: 4, neuron: 4342, label: "token counter", description: "counting repeated occurrences of a token"},
    {layer: 5, neuron: 2326, label: "rhymes with 'at'", description: "syllables rhyming with 'at', sometimes 'it', 'et', 'ot'"},
    {layer: 5, neuron: 4492, label: "leading 'an'", description: "sentences that start with 'an'"}, // score 0.77
    {layer: 6, neuron: 3251, label: "not all", description: "not all"},
    {layer: 10, neuron: 2851, label: "leading acronyms", description: "acronyms after punctuation or newlines"},
    {layer: 12, neuron: 2884, label: "hypothetical had", description: "had in hypothetical contexts"}, // 
    {layer: 14, neuron: 3539, label: "long sequences", description: "long sequences of stuff"},
    {layer: 14, neuron: 3822, label: "X by/after *X*", description: "noun repetitions separated by 'by' or 'after'"},
    {layer: 21, neuron: 3982, label: "any *and* all", description: "any/anything *and/&* all/everything"},
    {layer: 26, neuron: 20, label: "truth, skin, or sun", description: "truth, skin, or sun"},
    // layer=18&neuron=5302
    /**************
    /* boring
    ***************/
    /**************
    /* poorly explained + interesting
    ***************/
    {heading: 'Poorly explained by GPT-4', layer: 0, neuron: 0, label: ''},
    // Actually activates for negated version “not so much … as” even when not so much is fairly far apart
    // another "not all":  13&neuron=1352
    // {layer: 0, neuron: 2823, label: "Hillary email leak vibes", description: "contexts related to Hillary Clinton leaked emails"}, // score ??
    // {layer: 12, neuron: 3718, label: "comparative phrases and negations", description: "comparative phrases and negations"}, // score 0.12
    {layer: 13, neuron: 410, label: "N and N+1", description: "a number following its predecessor"}, // score ??
    {layer: 13, neuron: 979, label: "subtle plurals", description: "subtle/nonobvious plurals"}, // score ??
    // slash after number 12&neuron=847
    // numbers predicting slash: 14&neuron=92
    // 0&neuron=2823
    {layer: 14, neuron: 1251, label: "subjunctive verbs", description: "verbs in subjunctive mood"}, // score ??
    {layer: 16, neuron: 518, label: "pattern breaks", description: "tokens that break an established pattern in an ongoing list"}, // score 0.2 with totally wrong explanation
    {layer: 17, neuron: 821, label: "idioms", description: "idioms"},
    {layer: 18, neuron: 3481, label: "post-typo", description: "first token following a typo"}, // score ??
    {layer: 18, neuron: 3552, label: "repeated text", description: "repeated text"}, // score ??
    // another shared last names: https://openaipublic.blob.core.windows.net/neuron-explainer/neuron-viewer/index.html#/layers/20/neurons/3164
    {layer: 19, neuron: 1763, label: "shared last names", description: "last names when two different people sharing last name are mentioned"}, // score 0.36
    {layer: 20, neuron: 4334, label: "previous break", description: "tokens that previously preceded a linebreak"}, // score ??
    {layer: 27, neuron: 116, label: "MTG vibes", description: "Magic the Gathering contexts"}, // score ??
    {layer: 35, neuron: 1523, label: "NBA name predictor", description: "NBA person/player name predictor"}, // score ??
    // {layer: 36, neuron: 2275, label: "she predictor", description: "prediction of the token 'she'"}, // score ??
    // {layer: 36, neuron: 5107, label: "Mormon vibes", description: "Mormon related context"}, // score ??
    // ] predictor 40&neuron=4505
    {layer: 46, neuron: 2181, label: "C predictor", description: "prediction of the token 'C'"}, // score ??
  ]

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    navigate(`/layers/${input_layer}/neurons/${input_neuron}`)
    return false
  }

  const handleNeuronClick = (layer: number, neuron: number) => {
    navigate(`/layers/${layer}/neurons/${neuron}`)
  }

  const feelingLuckySubmit = () => {
    const layer = Math.floor(Math.random() * 48);
    const neuron = Math.floor(Math.random() * 6400);
    navigate(`/layers/${layer}/neurons/${neuron}`)
    return false
  }


  return (
    <div className="flex flex-col items-center justify-center">
      <h1 className="text-2xl font-bold mb-4">Welcome!  Pick a neuron:</h1>
      <form
        onSubmit={handleSubmit}
        className="flex flex-col items-center justify-center"
        style={{ flexFlow: 'row wrap' }}
      >
        Layer <input
          type="number"
          id="inputLayer"
          value={input_layer}
          min={0}
          max={47}
          style={{ width: 70, marginLeft: 10, marginRight: 10 }}
          onChange={(e) => setLayer(parseInt(e.target.value))}
          className="border border-gray-300 rounded-md p-2"
        />
        Index <input
          type="number"
          id="inputNeuron"
          value={input_neuron}
          min={0}
          max={6399}
          style={{ width: 70, marginLeft: 10, marginRight: 10 }}
          onChange={(e) => setNeuron(parseInt(e.target.value))}
          className="border border-gray-300 rounded-md p-2"
        />
        <button
          onClick={handleSubmit}
          className="border border-gray-300 rounded-md p-2 mb-4 mt-4"
        >
          Go to {input_layer}:{input_neuron}
        </button>
      </form>
      <button
        onClick={feelingLuckySubmit}
        className="border border-gray-300 rounded-md p-2 mb-4 mt-4"
      >
        I'm feeling lucky
      </button>
      <div className="mt-4">
        <h2 className="text-xl font-bold mb-2">Interesting neurons:</h2>
        <div className="mb-10 flex-row">
          <div
            className="flex flex-flow flex-wrap"
          >
            {knownGoodNeurons.map(({ heading, layer, neuron, label, description }, j) => (
              heading ? <div style={{width: '100%'}} key={j}><h4>
              {heading}
              </h4></div> : <button
                onClick={() => handleNeuronClick(layer, neuron)}
                key={`${layer}:${neuron}`}
                style={{ width: 200 }}
                className="m-2 text-blue-500 hover:text-blue-700"
                title={description}
              >
                {label} ({layer}:{neuron})
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default NeuronForm
