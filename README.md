
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

# أسعار استخدام النماذج اللغوية

إذا سبق لك وتعاملت مع API مدفوعة، في الغالب أن السعر مرتبط بعدد استخدام ال API. لكن النماذج اللغوية يختلف الأمر؛ فالسعر ليس على عدد الاستخدام، بل على حجم رموز المدخلات والمخرجات. لكن ماهو الرمز؟ النماذج اللغوبة تقوم بترميز الكلمات قبل معالجتها. والترميز ممكن تبسيطة الى تقسيم النصوص الى أجزاء صغيرة (كلمات أو حروف). مايهمنا هنا أن اللغة العربية للأسف يتم ترميزها في الغالب إلى حروف بينما اللغة الإنجليزية إلى كلمات. هذا يعني أن استخدام النماذج اللغوية الحالية باللغة العربية سيكلفك من ثلاث إلى اربع اضعاف سعر الأستخدام باللغة الإنجليزية. أنظر للصورتين أدناه كيف أن ChatGPT 4 يقوم بترميز نص بالعربي بثلاث اضعاف الترميز الازم للغة الإنجليزية لنص مساوي في عدد الأحرف.


![token English](https://raw.githubusercontent.com/Alaboudi1/llmBenchmarkArabic/main/tokenizerEn.png "token English")

![token Arabic](https://raw.githubusercontent.com/Alaboudi1/llmBenchmarkArabic/main/tokenizerAr.png "token Arabic")

في هذا الجدول يعرض اسعار النماذج لكل مليون رمز:

| النموذج            | السعر لكل مليون رمز مدخل (دولار) | السعر لكل مليون رمز مخرج (دولار)       |
|--------------------|----------------------------------|----------------------------------------|
| Llama3.1_8b        | $0.06                            | $0.08                                  |
| o4mini             | $0.15                            | $0.60                                  |
| Gemini1.5-flash (< 128K token)  | $0.35               | $1.05 
| Gemini1.5-flash (> 128K token)   | $0.70               |  $2.10                                |
| mistral-nemo       | $0.30                            | $0.30                                  |
| Aya-8b             | $0.20 (تقديري)                  | $0.40 (تقديري)                          |
| Llama3.1_70b       | $0.59                            | $0.79                                  |
| Command-R          | $0.50                            | $1.50                                  |
| Gemma-2-27B        | $0.80                            | $0.80                                  |
| Aya-35b            | $1.00 (تقديري)                  | $3.00 (تقديري)                          |
| Llama3.1_405b      | $2.80                            | $2.80                                  |
| Large              | $3.00                            | $9.00                                  |
| claude-3.5-sonnet  | $3.00                            | $15.00                                 |
| Command-R+         | $3.00                            | $15.00                                 |
| Gemini1.5-pro   (< 128K token)   | $3.50              | $10.50                                 |
| Gemini1.5-pro   (> 128K token)   | $7                 | $12                                    |
| o4                 | $5.00                            | $15.00                                 |


فعلى سبيل المثال لو أرسلت هذا المدخل للنموذج ChatGPT o4:
- **المستخدم:** السلام عليكم
- **النموذج:** وعليكم السلام ورحمة الله وبركاته

فالمدخل "السلام عليكم" يساوي ٩ رموز والمخرج "وعليكم السلام ورحمة الله وبركاته" يساوي 23 رمز. المدخل كلف ٠.٠٠٠٠٤٥ دولار، والمخرج كلف ٠.٠٠٠٣٤٥ دولار، فالسعر النهائي لهذه المحادثة هو ٠.٠٠٠٣٩ دولار أو ٠.٠٠١٥ ريال.
