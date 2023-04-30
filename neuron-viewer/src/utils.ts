export const memoizeAsync = (fnname: string, fn: any) => {
  return async (...args: any) => {
    const key = `memoized:${fnname}:${args.map((x: any) => JSON.stringify(x)).join("-")}`
    const val = localStorage.getItem(key);
    if (val === null) {
      const value = await fn(...args)
      localStorage.setItem(key, JSON.stringify(value))
      console.log(`memoized ${fnname}(${args.map((x: any) => JSON.stringify(x)).join(", ")})`, value)
      return value
    } else {
      // console.log(`parsing`, val)
      return JSON.parse(val)
    }
  }
}


export const getQueryParams = () => {
  const urlParams = new URLSearchParams(window.location.search)
  const params: {[key: string]: any} = {}
  for (const [key, value] of urlParams.entries()) {
    params[key] = value
  }
  return params
}
