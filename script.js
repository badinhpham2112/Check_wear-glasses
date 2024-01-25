const container = document.querySelector('#container');
const fileInput = document.querySelector('#file-input');

async function loadTrainingData() {
    const labels = ['Đeo kính', 'Không đeo kính'];
    const faceDescriptors = [];

    for (const label of labels) {
        const descriptors = [];
        for (let i = 1; i <= 10; i++) {
            const image = await faceapi.fetchImage(`/data/${label}/${i}.jpg`);
            const detectionsWithLandmarks = await faceapi.detectSingleFace(image).withFaceLandmarks()
            if (detectionsWithLandmarks) {
                const descriptorsSingleFace = await faceapi.computeFaceDescriptor(image, detectionsWithLandmarks.landmarks);
                descriptors.push(descriptorsSingleFace);
            }
        }
        faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
        Toastify({ text: `Training xong dữ liệu của ${label}` }).showToast();
    }
    return new faceapi.FaceMatcher(faceDescriptors);
}

let faceMatcher;
async function init() {
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models')
    ]);

    const trainingData = await loadTrainingData();
    const labeledDescriptorsArray = trainingData.labeledDescriptors.map(ld => new faceapi.LabeledFaceDescriptors(ld.label, ld.descriptors));
    faceMatcher = await new faceapi.FaceMatcher(labeledDescriptorsArray, 0.6);
    // faceMatcher = await new faceapi.FaceMatcher(trainingData, 0.6);
    Toastify({ text: "Tải xong model nhận diện!" }).showToast();
}

init();

fileInput.addEventListener('change', async(e) => {
    const file = fileInput.files[0];
    const image = await faceapi.bufferToImage(file);
    const canvas = faceapi.createCanvasFromMedia(image);
    container.innerHTML = '';
    container.append(image);
    container.append(canvas);
    const size = {
        width: image.width,
        height: image.height
    };
    faceapi.matchDimensions(canvas, size);
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, size);
    console.log('Detected Faces:', resizedDetections);

    for (const singleDetection of resizedDetections) {
        // Check if a face is detected
        if (singleDetection && singleDetection.detection) {
            const descriptors = await faceapi.computeFaceDescriptor(image, singleDetection.detection.landmarks);

            // Log descriptors and labels
            console.log('Descriptors:', descriptors);
            console.log('Labels:', faceMatcher.labeledDescriptors.map(ld => ld.label));

            // Check if descriptors are available
            if (descriptors.length > 0) {
                const bestMatch = faceMatcher.findBestMatch(descriptors);
                const drawBox = new faceapi.draw.DrawBox(singleDetection.detection.box, {
                    label: bestMatch.label
                });
                drawBox.draw(canvas);

                // Log the label of the best match
                console.log('Best Match Label:', bestMatch.label);
            }
        }
    }

});