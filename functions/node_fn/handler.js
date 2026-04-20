const express = require('express');
const os = require('os');
const app = express();
app.use(express.json());

app.get('/', (req, res) => {
    res.json({ status: 'ok', runtime: 'nodejs', host: os.hostname() });
});

app.post('/', (req, res) => {
    const result = Math.random() * 1000;
    res.json({ result, runtime: 'nodejs' });
});

app.listen(8080);
