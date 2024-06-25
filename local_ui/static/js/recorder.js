// set up basic variables for app
const start = document.querySelector('.starter');
const stop = document.querySelector('.stopper');
const canvas = document.querySelector('.visualizer');
const mainSection = document.querySelector(".mainSection");
var dispText = document.querySelector(".disp");
var inConvo = false;
var inDecode = false;
var backBuffer = [];
var chirpyBuffer = [];
let firstutt = true;
const audio = document.querySelector('audio');

audio.addEventListener("ended", async function () {
    audio.currentTime = 0;
    console.log("ended");
    socket.emit("agent-said", false);
});

// audio.addEventListener("ended", async function () {
//     audio.currentTime = 0;
//     console.log("ended");
//     socket.emit("agent-said", false);
//     let found = false;
//     if (chirpyBuffer.length > 0) {
//         while ((!found) && chirpyBuffer.length > 0) {
//             let arrEnt = chirpyBuffer[0];
//             console.log("Current chirpyBuffer", chirpyBuffer);
//             await new Promise(r => {
//                 setTimeout(() => {
//                     let currTime = Date.now();
//                     console.log(currTime, arrEnt[0])
//                     if (((currTime - arrEnt[0] >= 0) && (currTime - arrEnt[0] < 1500))) {
//                         found = true;
//                         arrEnt = chirpyBuffer.shift();
//                         socket.emit("agent-said", true);
//                         audio.load();
//                         audio.src = arrEnt[1];
//                         console.log("Playing audio");
//                         audio.play();
//                         arrEnt = chirpyBuffer.shift();
//                     } else if (currTime - arrEnt[0] >= 1500) {
//                         socket.emit("agent-said", false);
//                         arrEnt = chirpyBuffer.shift();
//                     }
//                 }, 10);
//             });
//
//         }
//     }
//     if ((backBuffer.length > 0) && (!found)) {
//         while ((!found) && backBuffer.length > 0) {
//             let arrEnt = backBuffer[0];
//             console.log("Current agentBuffer", backBuffer);
//             await new Promise(r => {
//                 setTimeout(() => {
//                     let currTime = Date.now();
//                     console.log(currTime, arrEnt[0])
//                     if (((currTime - arrEnt[0] > 0) && (currTime - arrEnt[0] < 1500)) || (arrEnt[2] == -1)) {
//                         found = true;
//                         arrEnt = backBuffer.shift();
//                         socket.emit("agent-said", true);
//                         audio.load();
//                         audio.src = arrEnt[1];
//                         console.log("Playing audio");
//                         audio.play();
//                         backBuffer = [];
//                         arrEnt = backBuffer.shift();
//                     } else if (currTime - arrEnt[0] >= 1500) {
//                         socket.emit("agent-said", false);
//                         arrEnt = backBuffer.shift();
//                     }
//                 }, 10);
//             });
//
//         }
//     }
// });

function getScriptPath(foo) {
    return window.URL.createObjectURL(new Blob([foo.toString().match(/^\s*function\s*\(\s*\)\s*\{(([\s\S](?!\}$))*[\s\S])/)[1]], {type: 'text/javascript'}));
}


var socket = io.connect('http://localhost:5023');
socket.on('connect', function () {
    console.log("Connected...!", socket.connected);
});

// disable stop button while not recording

stop.disabled = true;

// visualiser setup - create web audio api context and canvas

let audioCtx;
const canvasCtx = canvas.getContext("2d");

//main block for doing the audio recording

if (navigator.mediaDevices.getUserMedia) {
    console.log('getUserMedia supported.');
    const constraints = {audio: true};

    let onSuccess = function (stream) {
        const mediaRecorder = RecordRTC(stream, {
            type: 'audio',
            mimeType: 'audio/wav',
            desiredSampRate: 16000, // accepted sample rate by Azure
            // sampleRate: 16000,
            timeSlice: 500,
            ondataavailable: (blob) => {
                if (inConvo) {
                    socket.emit('user-audio', blob); // sends blob to server
                    console.log("sent blob");
                }
            },
            recorderType: StereoAudioRecorder,
            numberOfAudioChannels: 1
        })

        visualize(stream);

        start.onclick = function () {
            inConvo = true;
            mediaRecorder.startRecording();
            console.log(mediaRecorder.state);
            console.log("recorder started");
            start.style.background = "red";

            stop.disabled = false;
            start.disabled = true;
            socket.emit("start-convo", "started conversation");
        }

        stop.onclick = function () {
            inConvo = false;
            mediaRecorder.stopRecording();
            console.log(mediaRecorder.state);
            console.log("recorder stopped");
            start.style.background = "";
            start.style.color = "";
            // mediaRecorder.requestData();

            audio.pause();
            audio.currentTime = 0;

            stop.disabled = true;
            start.disabled = false;
            backBuffer = [];
            socket.emit("stop-convo", "stopped");
        }


        // When the client receives a voice message it will play the sound
        socket.on('agent-audio', async function (byteArray) {
            console.log("Received:", Date.now());
            initAudio(byteArray);
        });


        socket.on('user-asr', function (txt) {
            console.log(txt);
            dispText.innerText = txt;
        });

    }

    let onError = function (err) {
        console.log('The following error occured: ' + err);
    }

    navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);

} else {
    console.log('getUserMedia not supported on your browser!');
}

async function initAudio(buffer) {
    // show audio player
    const blob = new Blob([buffer], {type: 'audio/mp3'})
    // const blob = new Blob([wavBytes], {type: 'audio/mp3'})
    let u = URL.createObjectURL(blob);
    console.log("Audio Current Time", audio.currentTime);
    socket.emit("agent-said", true);
    // audio.load();
    audio.src = u;
    console.log("Playing audio");
    // Show loading animation.
    var playPromise = audio.play();

    if (playPromise !== undefined) {
        playPromise.then(_ => {
            // Automatic playback started!
            // Show playing UI.
        })
            .catch(error => {
                // Auto-play was prevented
                // Show paused UI.
            });
    }
    // audio.play();


    // worker.postMessage(agentBuffer)
}

// adapted from https://gist.github.com/also/900023
function buildWaveHeader(opts) {
    const numFrames = opts.numFrames;
    const numChannels = opts.numChannels || 1;
    const sampleRate = opts.sampleRate || 16000;
    const bytesPerSample = opts.bytesPerSample || 2;
    const format = opts.format

    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numFrames * blockAlign;

    const buffer = new ArrayBuffer(44);
    const dv = new DataView(buffer);

    let p = 0;

    function writeString(s) {
        for (let i = 0; i < s.length; i++) {
            dv.setUint8(p + i, s.charCodeAt(i));
        }
        p += s.length;
    }

    function writeUint32(d) {
        dv.setUint32(p, d, true);
        p += 4;
    }

    function writeUint16(d) {
        dv.setUint16(p, d, true);
        p += 2;
    }

    writeString('RIFF');              // ChunkID
    writeUint32(dataSize + 36);       // ChunkSize
    writeString('WAVE');              // Format
    writeString('fmt ');              // Subchunk1ID
    writeUint32(16);                  // Subchunk1Size
    writeUint16(format);              // AudioFormat
    writeUint16(numChannels);         // NumChannels
    writeUint32(sampleRate);          // SampleRate
    writeUint32(byteRate);            // ByteRate
    writeUint16(blockAlign);          // BlockAlign
    writeUint16(bytesPerSample * 8);  // BitsPerSample
    writeString('data');              // Subchunk2ID
    writeUint32(dataSize);            // Subchunk2Size

    return buffer;
}


function visualize(stream) {
    if (!audioCtx) {
        audioCtx = new AudioContext();
    }

    const source = audioCtx.createMediaStreamSource(stream);

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    source.connect(analyser);
    //analyser.connect(audioCtx.destination);

    draw()

    function draw() {
        const WIDTH = canvas.width
        const HEIGHT = canvas.height;

        requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = 'rgb(0, 0, 0)';
        canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

        canvasCtx.lineWidth = 3;
        canvasCtx.strokeStyle = 'rgb(255, 255, 255)';

        canvasCtx.beginPath();

        let sliceWidth = WIDTH * 4.0 / bufferLength;
        let x = 0;

        var sum = 0;

        for (let i = 0; i < bufferLength; i++) {

            let v = dataArray[i] / 128.0;
            let y = v * HEIGHT / 2;

            sum = sum + ((dataArray[i] - 128.0) & 0xFF);

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();

    }
}

window.onresize = function () {
    canvas.width = mainSection.offsetWidth;
}

window.onresize();