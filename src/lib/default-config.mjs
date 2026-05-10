const PROJECT_ID = "bustchester-e08c3";
const REGION = "us-central1";
const FUNCTIONS_BASE_URL = `https://${REGION}-${PROJECT_ID}.cloudfunctions.net`;

export const BURSTCHESTER_DEFAULTS = Object.freeze({
  projectId: PROJECT_ID,
  region: REGION,
  firebaseConfig: Object.freeze({
    apiKey: "AIzaSyBT48mVt9IDw6Ctf_VjNl0JNc4S1SrVZfs",
    authDomain: "bustchester-e08c3.firebaseapp.com",
    projectId: PROJECT_ID,
    storageBucket: "bustchester-e08c3.firebasestorage.app",
    messagingSenderId: "542098071019",
    appId: "1:542098071019:web:99a5d7b6592d67514ecdae",
    measurementId: "G-XLYZE8H5LF",
  }),
  functionsBaseUrl: FUNCTIONS_BASE_URL,
  healthCheckUrl: `${FUNCTIONS_BASE_URL}/healthCheck`,
  profileUrl: `${FUNCTIONS_BASE_URL}/upsertCliProfile`,
  accessTokenUrl: `${FUNCTIONS_BASE_URL}/issueAccessToken`,
  datasetDownloadUrl: `${FUNCTIONS_BASE_URL}/prepareDatasetDownload`,
  modelDownloadUrl: `${FUNCTIONS_BASE_URL}/recordModelDownload`,
  modelRegisterUrl: `${FUNCTIONS_BASE_URL}/registerModelHttp`,
  pointCostUpdateUrl: `${FUNCTIONS_BASE_URL}/updateAssetPointCost`,
  debugUploadUrl: `${FUNCTIONS_BASE_URL}/debugUploadDataset`,
});
