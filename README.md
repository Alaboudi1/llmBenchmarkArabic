
![llmArabic](https://raw.githubusercontent.com/Alaboudi1/llmBenchmarkArabic/main/llmArbic.png "llmArabic")

# الذكاء الاصطناعي التوليدي باللغة العربية

في مارس 2023، أطلقت OpenAI أولى واجهات برمجة التطبيقات (APIs) الموجهة للمبرمجين لبناء تطبيقات باستخدام الذكاء الاصطناعي التوليدي (النماذج اللغوية الكبيرة). مع هذا الحدث، وُلد مجال *هندسة الأنظمة الذكية* (AI Engineering)، وهو مجال مختلف عن *هندسة تعلم الآلة* (ML Engineers). يميل مجال هندسة تعلم الآلة إلى الطابع البحثي ويهتم بكافة أنواع الذكاء الاصطناعي، بينما يميل مجال هندسة الذكاء الاصطناعي إلى التطبيق العملي، وهو في الأساس خليط من هندسة البرمجيات وبناء واستخدام نماذج اللغة الكبيرة (LLMs). لا أريد الأستطراد في مهام مهندسي الذكاء الأصطناعي، لكن اذا كنت مهتم [أقرأ المزيد هنا](https://www.latent.space/p/ai-engineer).

وتزامن مع إطلاق الـAPI للنماذج اللغوية بناء برمجيات وشركات كثيرة تستخدم الذكاء الاصطناعي التوليدي. وأصبح الكثير من الشركات تنافس في تزويد المبرمجين بأحدث نماذج الذكاء الاصطناعي التوليدي. بحسب موقع Chatbot Arena، هناك ٨٤ نموذجًا يتنافسون على تقديم أفضل نماذج الذكاء الاصطناعي التوليدي، [للمزيد اطلع على الموقع](https://chat.lmsys.org).

لكن السؤال المهم يبقى: "ما هو أفضل نموذج ذكاء اصطناعي توليدي باللغة العربية واللهجة السعودية يمكن استخدامه لبناء تطبيقات محلية في السعودية؟" هذا السؤال تكرر بشكل كبير ولا يوجد حتى الآن جواب قاطع عليه، إلا تجارب شخصية. هناك بعض الاجتهادات مثل [OALL](https://huggingface.co/blog/leaderboard-arabic)، لكن مثل هذه الاجتهادات تميل أن تكون أكاديمية ولا يوجد، لحد علمي، اختبار للهجة السعودية، والتي أراها مهمة لبناء تطبيقات محلية بالذكاء الاصطناعي التوليدي.

في هذا المقال، سأقدم تقريرًا مبسطًا عن نماذج الذكاء الاصطناعي ودعمها للغة العربية واللهجة السعودية، وأسعارها، والاستضافة وخصوصية البيانات، وفي الأخير مستقبل اللغة العربية ونماذج الذكاء الاصطناعي التوليدي وأتمنى أن يكون هذا المقال مفيد لأي شخص أو شركة تريد الدخول في هذا المجال.

## نماذج الذكاء الاصطناعي التوليدي (النماذج اللغوية الكبيرة) المتوفرة للمبرمجين

هناك الكثير من النماذج لغوية (أو نماذج ذكاء اصطناعي توليدي فكلهما بنفس المعنى) المتوفرة للمبرمجين، لكن في هذه المقالة سنركز على أشهر ١٥ نموذج لغوي. سأقوم بسرد المعلومات العامة عن هذه النماذج، هذه المعلومات من عدة مصادر سأسردها في أخر المقال. المعلومات العامة تتمركز حول ثلاث محاول: **حجم السياق**، **أسعار الاستخدام**، **وسياسات الاستخدام**.

### حجم السياق (Context Window)

السياق هو مجموعة من المعلومات المقدمة كمدخل ضمن الأمر النصي (prompt) للنموذج اللغوي. فعلى سبيل المثال، بدل من أن تقول للنموذج اللغوي "من هو أحمد؟" وتتوقع منه الإجابة الصحيحة، تعطية لمحة تاريخية عن شخصية أحمد وسيرته الذاتية قبل طرح سؤالك. هناك فائدتان رئيسيتان للسياق في النماذج اللغوية الكبيرة:
1. **توفير معلومات إضافية:** يمكن أن يتضمن السياق معلومات غير معروفة للنموذج مثل أسماء وأسعار السلع في متجرك الخاص، الأخبار الجديدة، أو أي معلومة لم تكن متاحة في بيانات التدريب الخاصة بالنموذج. العديد من هذه النماذج يتم تدريبها حتى تاريخ معين يعرف بـ "cut-off knowledge". أي معلومة جديدة لم تكن ضمن معلومات التدريب يجب تزويدها كسياق للنموذج لتجنب الهلوسة (الهلوسة هنا تعني الإجابة أو الرد الغير صحيح).
2. **استخدامه كذاكرة:** يمكن للنموذج استخدام السياق كذاكرة. على سبيل المثال، إذا أخبرت النموذج بأن اسمك عبد العزيز، فسيتذكر الاسم ما دامت المحادثة ضمن حجم السياق المحدد. إذا تعدت المحادثة حجم السياق فلن يكون اسمك من ضمن المدخلات للنموذج اللغوي، مما يعني أن لو سئلت النموذج عن اسمك لاحقاً في المحادثة ففي الغالب ستكون الإجابة عبارة عن هلوسة (مثلاً سيقول اسمك محمد).

كلما زاد حجم السياق، كانت إمكانية بناء تطبيقات أفضل على النموذج. ومع ذلك، يجب على مهندس الذكاء الاصطناعي إدارة السياق بشكل فعال ومحاولة تقليصه قدر الإمكان لأن زيادة حجم السياق مكلفة وقد تسبب بطء في استجابة النموذج. كما أن زيادة حجم السياق يمكن أن تزيد من احتمالية "الهلوسة" لدى النموذج.

للمزيد حول أهمية إدارة السياق، يمكنك قراءة الورقة الشهيرة: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172).

هذا الجدول يوضج حجم السياق للأشهر ١٥ نموذج لغوي موجود حالياً:
| النموذج            | حجم السياق (token) |
|--------------------|---------------------|
| Gemini1.5-pro      | 1M                  |
| Gemini1.5-flash    | 1M                  |
| claude-3.5-sonnet  | 200K                |
| chatGPT-o4         | 128K                |
| Llama3.1_405b      | 128K                |
|chatGPT-o4mini      | 128K                |
| Command-R+         | 128K                |
| Llama3.1_70b       | 128K                |
| mistral-Large2     | 128K                |
| Command-R          | 128K                |
| Llama3.1_8b        | 128K                |
| mistral-nemo       | 128K                |
| Aya-35b            | 8K                  |
| Aya-8b             | 8K                  |
| Gemma-2-27B        | 8K                  |

### أسعار استخدام النماذج اللغوية

إذا سبق لك وتعاملت مع API مدفوعة، في الغالب أن السعر مرتبط بعدد استخدام ال API. لكن النماذج اللغوية يختلف الأمر؛ فالسعر ليس على عدد الاستخدام، بل على حجم رموز المدخلات والمخرجات. لكن ماهو الرمز؟ النماذج اللغوبة تقوم بترميز الكلمات قبل معالجتها. والترميز ممكن تبسيطة الى تقسيم النصوص الى أجزاء صغيرة (كلمات أو حروف). مايهمنا هنا أن اللغة العربية للأسف يتم ترميزها في الغالب إلى حروف بينما اللغة الإنجليزية إلى كلمات. هذا يعني أن استخدام النماذج اللغوية الحالية باللغة العربية سيكلفك من ثلاث إلى اربع اضعاف سعر الأستخدام باللغة الإنجليزية. أنظر للصورتين أدناه كيف أن ChatGPT 4 يقوم بترميز نص بالعربي بثلاث اضعاف الترميز الازم للغة الإنجليزية لنص مساوي في عدد الأحرف.


![token English](https://raw.githubusercontent.com/Alaboudi1/llmBenchmarkArabic/main/tokenizerEn.png "token English")

![token Arabic](https://raw.githubusercontent.com/Alaboudi1/llmBenchmarkArabic/main/tokenizerAr.png "token Arabic")

في هذا الجدول يعرض اسعار النماذج لكل مليون رمز:

| النموذج            | السعر لكل مليون رمز مدخل (دولار) | السعر لكل مليون رمز مخرج (دولار)       |
|--------------------|----------------------------------|----------------------------------------|
| Llama3.1_8b        | $0.06                            | $0.08                                  |
| chatGPT-o4mini     | $0.15                            | $0.60                                  |
| Gemini1.5-flash (< 128K token)  | $0.35               | $1.05 
| Gemini1.5-flash (> 128K token)   | $0.70               |  $2.10                                |
| mistral-nemo       | $0.30                            | $0.30                                  |
| Aya-8b             | $0.20 (تقديري)                  | $0.40 (تقديري)                          |
| Llama3.1_70b       | $0.59                            | $0.79                                  |
| Command-R          | $0.50                            | $1.50                                  |
| Gemma-2-27B        | $0.80                            | $0.80                                  |
| Aya-35b            | $1.00 (تقديري)                  | $3.00 (تقديري)                          |
| Llama3.1_405b      | $2.80                            | $2.80                                  |
| mistral-Large2               | $3.00                            | $9.00                                  |
| claude-3.5-sonnet  | $3.00                            | $15.00                                 |
| Command-R+         | $3.00                            | $15.00                                 |
| Gemini1.5-pro   (< 128K token)   | $3.50              | $10.50                                 |
| Gemini1.5-pro   (> 128K token)   | $7                 | $12                                    |
| chatGPT-o4                   | $5.00                            | $15.00                                 |


فعلى سبيل المثال لو أرسلت هذا المدخل للنموذج ChatGPT o4:
- **المستخدم:** السلام عليكم
- **النموذج:** وعليكم السلام ورحمة الله وبركاته

فالمدخل "السلام عليكم" يساوي ٩ رموز والمخرج "وعليكم السلام ورحمة الله وبركاته" يساوي 23 رمز. المدخل كلف ٠.٠٠٠٠٤٥ دولار، والمخرج كلف ٠.٠٠٠٣٤٥ دولار، فالسعر النهائي لهذه المحادثة هو ٠.٠٠٠٣٩ دولار أو ٠.٠٠١٥ ريال.

### سياسة الاستخدام لنماذج اللغوية الكبيرة:
واحدة من أهم الأمور التي يجب على مهندسي الذكاء الاصطناعي الانتباه لها هي سياسة الاستخدام للنماذج اللغوية الكبيرة. فالكثير من النماذج تصنف تحت (مفتوحة الأوزان) بمعنى انه بمقدورك كمهندس تحميل النموذج في جهازك الخاص واستخدامه. لكن غالب تلك النماذج المفتوحة لا تستطيع استخدامها بشكل تجاري قبل شراء ترخيص من الشركة.
الجدول التالي يفصل سياسات الترخيص والاستضافة

| Model             | Open Wight | Private Hosting | Cloud Hosting                                                                                                                   | License Type                                                                 |
|-------------------|------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| claude-3.5-sonnet | ❌    | ❌      | • Google Cloud’s Vertex AI (Dammam, Saudi Arabia) <br> • Amazon Bedrock (Non-Saudi Hosting) <br> • Anthropic API (Non-Saudi Hosting) | Proprietary <br> [Anthropic Terms](https://www.anthropic.com/legal/commercial-terms) |
| o4                | ❌    | ❌            | • OpenAI API (Non-Saudi Hosting) <br> • Azure OpenAI Service (Non-Saudi Hosting)                                                | Proprietary <br> [OpenAI Terms](https://openai.com/policies/business-terms/)   |
| Gemini1.5-pro     | ❌    | ❌               | • Google Cloud’s Vertex AI (Dammam, Saudi Arabia)                                                                               | Proprietary <br> [Google Gemini Terms](https://ai.google.dev/gemini-api/terms) |
| Llama3.1_405b     | ✅    | ✅     | Any provider with GPU. Please see [Llama Partners](https://llama.meta.com/docs/getting-the-models/405b-partners/)                | Llama 3.1 Community License Agreement <br> [License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) |
| o4mini            | ❌    | ❌          | • OpenAI API (Non-Saudi Hosting) <br> • Azure OpenAI Service (Non-Saudi Hosting)                                                | Proprietary <br> [OpenAI Terms](https://openai.com/policies/business-terms/)   |
| Command-R+        | ✅      | • non-commercial use is free <br> • commercial needs License | • Google Cloud’s Vertex AI (Non-Saudi Hosting) <br> • Amazon Bedrock (Non-Saudi Hosting) <br> • Azure OpenAI Service (Non-Saudi Hosting) <br> • Cohere API (Non-Saudi Hosting) | Creative Commons Attribution Non Commercial 4.0 |
| Gemini1.5-flash   | ❌    | ❌              | • Google Cloud’s Vertex AI (Dammam, Saudi Arabia)                                                                               | Proprietary <br> [Google Gemini Terms](https://ai.google.dev/gemini-api/terms) |
| Llama3.1_70b      | ✅    | ✅     | Any provider with GPU. Please see [Llama Partners](https://llama.meta.com/docs/getting-the-models/405b-partners/)                | Llama 3.1 Community License Agreement <br> [License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) |
| Large             | ✅      | • non-commercial use is free <br> • commercial needs License | • Google Cloud’s Vertex AI (Non-Saudi Hosting) <br> • Amazon Bedrock (Non-Saudi Hosting) <br> • Azure OpenAI Service (Non-Saudi Hosting) <br> • Mistral API (Non-Saudi Hosting) | Mistral AI Research License <br> [License](https://mistral.ai/licenses/MRL-0.1.md) |
| Aya-35b           | ✅    | Only for non-commercial use |  • Google Cloud’s Vertex AI (Non-Saudi Hosting) <br> • Amazon Bedrock (Non-Saudi Hosting) <br> • Azure OpenAI Service (Non-Saudi Hosting) <br> • Cohere API (Non-Saudi Hosting) | Creative Commons Attribution Non-Commercial 4.0 |
| Aya-8b            | ✅    | Only for non-commercial use |  • Google Cloud’s Vertex AI (Non-Saudi Hosting) <br> • Amazon Bedrock (Non-Saudi Hosting) <br> • Azure OpenAI Service (Non-Saudi Hosting) <br> • Cohere API (Non-Saudi Hosting) | Creative Commons Attribution Non-Commercial 4.0 |
| Command-R         | ✅    | Only for non-commercial use |  • Google Cloud’s Vertex AI (Non-Saudi Hosting) <br> • Amazon Bedrock (Non-Saudi Hosting) <br> • Azure OpenAI Service (Non-Saudi Hosting) <br> • Cohere API (Non-Saudi Hosting) | Creative Commons Attribution Non-Commercial 4.0 |
| Llama3.1_8b       | ✅    | ✅     | Any provider with GPU. Please see [Llama Partners](https://llama.meta.com/docs/getting-the-models/405b-partners/)                | Llama 3.1 Community License Agreement <br> [License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) |
| Gemma-2-27B       | ✅    | ✅     | Any provider with GPU. Please see [Gemma Terms](https://huggingface.co/google/gemma-2-27b-it)                                    | Gemma Terms of Use <br> [Terms](https://ai.google.dev/gemma/terms) |
| mistral-nemo      | ✅    | ✅     | Any provider with GPU. Please see [Mistral Deployment](https://mistral.ai/technology/#deployment)                               | Apache License <br> Version 2.0 |

## أختبار القدرات العربية واللهجة السعودية لدى النماذج اللغوية الكبيرة
تكلمت مع أشخاص كثر لديهم أهتمام بإستخدام نماذج اللغوية الكبيرة في بناء وتحسين تطبيقات برمجية. في الغالب الأشخاص هؤلاء ينقسمون إلى قسمين: قسم يطرح السؤال "ماهو أفضل نموذج لغوي من ناحية الأداء في اللغة العربية؟" والاخر يستخدم chatGPT-o4. كان واضح عندي الحاجة لتقديم جاوب للسؤال المطروح ومحاولة تحدي الرأي السائد أن chatGPT وهو أفضل خيار. لهذا قررت عمل هذا الأختبار الأول من نوعة على حد علمي. الأختبار ينقسم إلى قسمين: القسم الأول عبارة عن ٤١٨ سؤال قياس لفظي تم سحبها من الأنترنت وتنظيفها وطلب من أشهر ١٥ نموذج لغوي للإجابة عليها. اسئلة اختبار قياس كما هو معلوم عبارة أختيارات متعددة. جميع الأسئلة متوفرة هنا للإطلاع. القسم الثاني من الأحتبار عبارة ثلاث أسئلة تحاكي الأسئلة التي تطرح على خدمة العملاء في القطاع العقاري. الهدف هو أختبار النماذج اللغوية على الاجابة على أسئلة العميل باللهحة السعودية. أستخدمت خليط من [LLM-as-a-Judge](https://arxiv.org/pdf/2306.05685) و Human-as-aJudge. القضاة من ناحية النماذج اللغوية الكبيرة، تم أختيار chatGPT-o4 و claude-3.5-sonnet. وأنا مثلت القضاة البشر. كل رد على الثلاث أسئلة تم تقييمة من ١٠. في الاسفل هي نتائج القسم الأول والثاني من الأختبار. 
