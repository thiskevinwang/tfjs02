require("dotenv").config();

const {
  DocumentUnderstandingServiceClient,
} = require("@google-cloud/documentai");
const uuid = require("uuid");
const fs = require("fs");
const path = require("path");

// const filePath = path.join(__dirname, "United-front.pdf");
const filePath = path.join(__dirname, "pdfs/Kevin.pdf");
const united = fs.readFileSync(filePath);

const projectId = process.env.PROJECT_ID;
const _location = process.env.LOCATION; // Format is 'us' or 'eu'
const bucket = process.env.BUCKET;
// const autoMLModel = 'Full resource name of AutoML Natural Language model';
// const gcsInputUri = "gs://cloud-samples-data/documentai/invoice.pdf";

const client = new DocumentUnderstandingServiceClient();

/**
 * # Run with:
 * ```sh
 * node documentai.js
 * ```
 */
async function quickstart() {
  const gcsOutputUriPrefix = uuid.v4();
  // Recognizes text entities in the PDF document
  const [result] = await client.processDocument({
    parent: `projects/${projectId}/locations/${_location}`,
    inputConfig: {
      // gcsSource: {
      //   uri: gcsInputUri,
      // },
      contents: united,
      /**
       * At this time, the only MIME types supported are 'application/pdf','application/json', 'image/gif' and 'image/tiff'.
       */
      mimeType: "application/pdf",
    },
    // outputConfig: {
    //   gcsDestination: {
    //     uri: `gs://${bucket}/${gcsOutputUriPrefix}/`,
    //   },
    //   pagesPerShard: 1,
    // },
    formExtractionParams: {
      // enabled: true,
      // valueTypes: string[] (optional)
      // ADDRESS, LOCATION, ORGANIZATION, PERSON, PHONE_NUMBER, ID,
      // NUMBER, EMAIL, PRICE, TERMS, DATE, NAME
      // keyValuePairHints: [
      //   {
      //     key: "Rx Bin",
      //     valueTypes: ["NUMBER"],
      //   },
      // ],
    },
    // automlParams: {
    //   /** 'projects/{project_id_or_number=*}/locations/{location_id=*}/models/{model_id_with_prefix=*}' */
    //   model: `projects/${projectId}/locations/${_location}/models/my_first_model`,
    // },
  });

  // Get all of the document text as one big string
  const { text } = result;

  // Extract shards from the text field
  function extractText(textAnchor) {
    // First shard in document doesn't have startIndex property
    const startIndex = textAnchor.textSegments[0].startIndex || 0;
    const endIndex = textAnchor.textSegments[0].endIndex;

    return text.substring(startIndex, endIndex);
  }

  for (const entity of result.entities) {
    console.log(`\nEntity text: ${extractText(entity.textAnchor)}`);
    console.log(`Entity type: ${entity.type}`);
    console.log(`Entity mention text: ${entity.mentionText}`);
    console.log(`Confidence: ${entity.confidence}`);
  }
}
quickstart();
