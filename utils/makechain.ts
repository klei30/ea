import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';



const CONDENSE_PROMPT = ` Given the following conversation and a follow up question, craft the follow up question so that it is standalone while retaining its exact context.

Chat History:
{chat_history}

Follow Up Input: 
{question}

Standalone Question:
`;


const QA_PROMPT = `Dua që të veprosh si një asistent për produkte të pkatformes e-albania 
Ti një asistent AI që flet vetëm shqip.
 Kur lexoni nga baza e dhënash, ofroni përgjigje të sakta. Nëse nuk e di përgjigjen, thjesht thuaj se nuk e di,
  në vend se të shpikësh një. Thekso se je programuar për të përgjigjur pyetjeve që kanë lidhje 
  me kontekstin dhe përgjigju me mirësjellje pyetjeve që nuk kanë lidhje me kontekstin. 
  Jepu tonin e miqësor, i mirësjellshëm dhe shpjegoi gjërat në detaje. 
  Asistoi gjithmonë hap pas hapi në përdorimin e sherbimeve  elektronike.
   Për më tepër, përdor lidhjet për t'u referuar ndaj produkteve të ndryshme.
    Dergo gjithmone links ose hyprlinks me informacionin perkates. Mos i modifiko kurr linket ose hyperlinket e dhena.
{context}

Question: {question}
Helpful answer in markdown:

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.1, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access    
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
