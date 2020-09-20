# tfjs

## Findings

### Browser Code

This uses `webpack`

```sh
npm run start
```

### Server Code

This uses `tsc` & `node`

```sh
npm run dev
```

### Errors

```sh
npm i -D @types/webgl2
```

This fixes a webgl error when trying to compile with `tsc`

- https://github.com/tensorflow/tfjs/issues/2007
