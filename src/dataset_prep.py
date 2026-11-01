"""
Dataset Preparation Script for Medical Text Simplifier & Catastrophic Forgetting Study
Generates:
1. data/medical_train.jsonl (~650 examples)
2. data/medical_val.jsonl (75 examples)
3. data/medical_test.jsonl (75 examples)
4. data/general_eval_benchmark.jsonl (100 strictly held-out general prompts)
"""

import json
import os
import random

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Base Medical Knowledge Base (Templates and Clinical Jargon -> Plain Equivalents)
# -----------------------------------------------------------------------------
MEDICAL_ENTRIES = [
    # Cardiology
    ("The patient was diagnosed with acute myocardial infarction secondary to coronary artery thrombosis.",
     "The patient had a sudden heart attack caused by a blood clot blocking blood flow to the heart.", "Cardiology"),
    ("Echocardiogram revealed severe left ventricular hypertrophy with diminished ejection fraction.",
     "An ultrasound of the heart showed the main pumping chamber wall was severely thickened and pumping poorly.", "Cardiology"),
    ("Continuous telemetry monitoring demonstrated persistent paroxysmal atrial fibrillation with rapid ventricular response.",
     "Heart monitoring showed an irregular, very fast heartbeat coming from the upper chambers of the heart.", "Cardiology"),
    ("The physician initiated treatment for essential hypertension with an angiotensin-converting enzyme inhibitor.",
     "The doctor started treating the patient's high blood pressure with an ACE inhibitor pill to relax blood vessels.", "Cardiology"),
    ("Patient reports intermittent substernal chest discomfort exacerbated by physical exertion.",
     "The patient describes chest pain behind the breastbone that gets worse during physical activity.", "Cardiology"),
    ("Coronary angiography demonstrated high-grade stenosis within the proximal left anterior descending artery.",
     "A heart catheter test showed a severe blockage in one of the main blood vessels feeding the heart.", "Cardiology"),
    ("The patient developed acute decompensated congestive heart failure accompanied by bilateral lower extremity edema.",
     "The patient suffered a sudden flare-up of heart failure with significant swelling in both legs.", "Cardiology"),
    ("Cardiac auscultation revealed a harsh holosystolic murmur loudest at the cardiac apex.",
     "Listening to the chest with a stethoscope revealed a loud heart murmur over the tip of the heart.", "Cardiology"),
    ("The patient underwent successful percutaneous transluminal coronary angioplasty with drug-eluting stent placement.",
     "The patient had a successful procedure to open a blocked heart artery using a tiny balloon and a medicated mesh tube.", "Cardiology"),
    ("Electrocardiogram exhibited significant ST-segment elevation in precordial leads V1 through V4.",
     "An electrical heart tracing showed clear signs of an active heart attack affecting the front of the heart.", "Cardiology"),

    # Pulmonology
    ("Chest radiography demonstrated bilateral patchy infiltrates consistent with community-acquired pneumonia.",
     "A chest X-ray showed cloudy patches in both lungs typical of lung infection caught outside the hospital.", "Pulmonology"),
    ("The patient exhibits chronic obstructive pulmonary disease characterized by frequent acute exacerbations.",
     "The patient has long-term lung disease with frequent flare-ups of severe breathing difficulties.", "Pulmonology"),
    ("Computed tomography pulmonary angiography identified a saddle pulmonary embolism occluding both main pulmonary arteries.",
     "A specialized chest CT scan detected a large blood clot blocking the main arteries feeding both lungs.", "Pulmonology"),
    ("Spirometry demonstrated irreversible airflow limitation with an FEV1/FVC ratio below seventy percent.",
     "A lung breathing test showed permanent narrowing of the airways that does not fully open with inhalers.", "Pulmonology"),
    ("The patient reported progressive exertional dyspnea accompanied by persistent orthopnea.",
     "The patient had worsening shortness of breath during movement and difficulty breathing while lying flat.", "Pulmonology"),
    ("Arterial blood gas analysis revealed severe hypoxemia with compensated respiratory acidosis.",
     "A blood sample from the artery showed dangerously low oxygen levels and high acid levels from trapped carbon dioxide.", "Pulmonology"),
    ("High-resolution chest CT confirmed extensive bilateral idiopathic pulmonary fibrosis with honeycombing.",
     "A detailed lung scan confirmed widespread irreversible lung scarring with honeycomb-like damage.", "Pulmonology"),
    ("Thoracentesis was performed yielding straw-colored exudative pleural effusion.",
     "Fluid was drained from the space around the lung, showing inflammation or infection.", "Pulmonology"),
    ("The patient presented with spontaneous pneumothorax requiring emergent needle decompression.",
     "The patient arrived with a sudden collapsed lung that required urgent insertion of a needle to release trapped air.", "Pulmonology"),
    ("Physical examination noted diffuse expiratory wheezing and prolonged expiratory phase bilaterally.",
     "Listening to the lungs showed whistling breathing sounds and slow exhalation in both lungs.", "Pulmonology"),

    # Neurology
    ("Magnetic resonance imaging of the brain revealed an acute ischemic cerebrovascular accident in the middle cerebral artery territory.",
     "A brain MRI showed a recent stroke caused by blocked blood flow in a major brain artery.", "Neurology"),
    ("The patient presented with transient ischemic attack manifested by acute-onset hemiparesis and expressive aphasia.",
     "The patient had a mini-stroke causing sudden weakness on one side of the body and trouble speaking.", "Neurology"),
    ("Neurological assessment confirmed severe distal symmetrical sensorimotor polyneuropathy.",
     "A nerve exam confirmed nerve damage causing numbness, tingling, and weakness in both feet and hands.", "Neurology"),
    ("The patient suffers from refractory epilepsy characterized by complex partial seizures secondarily generalized.",
     "The patient has hard-to-control seizures that begin in one brain area and spread to shake the whole body.", "Neurology"),
    ("Lumbar puncture demonstrated elevated cerebrospinal fluid opening pressure and pleocytosis.",
     "A spinal tap test showed high fluid pressure and an increased number of white blood cells around the spinal cord.", "Neurology"),
    ("The patient exhibits Parkinsonian tremor, cogwheel rigidity, and progressive postural instability.",
     "The patient has rhythmic hand shaking, stiff ratchet-like limb movement, and trouble keeping balance.", "Neurology"),
    ("Non-contrast head CT excluded acute intracranial hemorrhage prior to intravenous thrombolysis.",
     "A head CT scan verified there was no bleeding inside the brain before giving clot-dissolving medicine.", "Neurology"),
    ("The patient complained of severe hemicranial throbbing cephalalgia with photophobia and phonophobia.",
     "The patient complained of intense pulsing headache on one side of the head with sensitivity to light and sound.", "Neurology"),
    ("Physical exam revealed positive Babinski sign and hyperreflexia indicative of upper motor neuron lesion.",
     "Reflex tests showed an abnormal toe reflex and overactive reflexes suggesting damage in the brain or spinal cord.", "Neurology"),
    ("Electroencephalography revealed generalized spike-and-wave discharges at three Hertz.",
     "A brain wave test showed abnormal rhythmic electrical spikes typical of absence seizures.", "Neurology"),

    # Gastroenterology & Hepatology
    ("Upper gastrointestinal endoscopy revealed severe erosive esophagitis and a bleeding duodenal ulcer.",
     "A camera exam of the stomach showed severe irritation of the food pipe and an ulcer bleeding in the small intestine.", "Gastroenterology"),
    ("Abdominal ultrasonography confirmed cholelithiasis with acoustic shadowing and acute calculous cholecystitis.",
     "An ultrasound of the belly confirmed gallstones and sudden, painful inflammation of the gallbladder.", "Gastroenterology"),
    ("The patient exhibits decompensated hepatic cirrhosis complicated by tense ascites and hepatic encephalopathy.",
     "The patient has advanced liver scarring leading to severe belly fluid buildup and confusion from toxin accumulation.", "Gastroenterology"),
    ("Colonoscopy demonstrated mucosal ulcerations and friability extending from the rectum, diagnostic of ulcerative colitis.",
     "A scope exam of the large intestine showed inflamed, easily bleeding sores characteristic of ulcerative colitis.", "Gastroenterology"),
    ("Serum amylase and lipase were markedly elevated, confirming acute interstitial pancreatitis.",
     "Blood tests measuring digestive enzymes were extremely high, confirming sudden inflammation of the pancreas.", "Gastroenterology"),
    ("The patient presented with massive hematemesis secondary to bleeding gastroesophageal varices.",
     "The patient vomited large amounts of blood due to burst swollen veins in the lower food pipe.", "Gastroenterology"),
    ("Percutaneous liver biopsy demonstrated bridging fibrosis and diffuse macrovesicular steatosis.",
     "A small tissue sample from the liver showed extensive scarring and heavy fat deposits inside liver cells.", "Gastroenterology"),
    ("The patient reported persistent dysphagia to solid foods and progressive involuntary weight loss.",
     "The patient had ongoing difficulty swallowing solid food and unintentionally lost significant body weight.", "Gastroenterology"),
    ("Intravenous proton pump inhibitors were administered to achieve intragastric acid suppression.",
     "Strong stomach acid-blocking medication was given through an IV to allow stomach ulcers to heal.", "Gastroenterology"),
    ("Barium swallow study demonstrated achalasia with failure of lower esophageal sphincter relaxation.",
     "A special X-ray swallowing test showed the muscle ring at the bottom of the food pipe cannot relax to let food pass.", "Gastroenterology"),

    # Nephrology & Urology
    ("The patient developed acute kidney injury manifested by anuria and elevated serum creatinine.",
     "The patient had sudden kidney failure, stopped producing urine, and had high waste levels in the blood.", "Nephrology"),
    ("Renal ultrasonography demonstrated bilateral hydronephrosis secondary to bladder outlet obstruction.",
     "A kidney ultrasound showed swelling in both kidneys caused by a blockage preventing the bladder from emptying.", "Nephrology"),
    ("Urinalysis revealed gross hematuria, heavy proteinuria, and red blood cell casts indicative of glomerulonephritis.",
     "A urine test showed visible blood, high protein, and clumped cells indicating inflammation in the kidney filters.", "Nephrology"),
    ("The patient requires chronic hemodialysis via a mature left radiocephalic arteriovenous fistula.",
     "The patient needs regular blood-filtering machine treatments through a surgically connected vein in the left arm.", "Nephrology"),
    ("Non-contrast helical CT identified an obstructing five millimeter calculus in the distal ureter.",
     "A quick CT scan found a small kidney stone blocking the tube that carries urine from the kidney to the bladder.", "Nephrology"),
    ("Estimated glomerular filtration rate decreased to fifteen milliliters per minute, signifying stage five chronic kidney disease.",
     "Kidney function testing dropped to fifteen percent of normal, indicating end-stage kidney failure.", "Nephrology"),
    ("The patient was treated for acute pyelonephritis with intravenous broad-spectrum beta-lactam antibiotics.",
     "The patient was treated for a severe bacterial kidney infection using powerful IV antibiotics.", "Nephrology"),
    ("The patient has diabetic nephropathy characterized by persistent microalbuminuria.",
     "The patient has kidney damage from long-term diabetes, detected by tiny amounts of protein leaking into the urine.", "Nephrology"),
    ("Cystoscopy visualized a papillary urothelial neoplasm along the posterior bladder wall.",
     "Looking inside the bladder with a camera revealed a small mushroom-like tumor on the back wall of the bladder.", "Nephrology"),
    ("Serum electrolyte panel revealed dangerous hyperkalemia requiring emergent administration of calcium gluconate.",
     "A blood test showed dangerously high potassium levels that required urgent medication to protect the heart rhythm.", "Nephrology"),

    # Oncology & Hematology
    ("Histopathological examination of the excised lymph node confirmed diffuse large B-cell lymphoma.",
     "Laboratory examination of the removed lymph gland confirmed a fast-growing cancer of the white blood cells.", "Oncology"),
    ("The bone marrow biopsy revealed acute myeloid leukemia with sixty percent myeloblasts.",
     "A bone marrow test showed blood cancer with sixty percent immature, abnormal blood-forming cells.", "Oncology"),
    ("The patient developed chemotherapy-induced febrile neutropenia requiring immediate hospitalization.",
     "Chemotherapy caused the patient's infection-fighting white blood cells to drop to dangerous levels alongside a high fever.", "Oncology"),
    ("Positron emission tomography scan revealed hypermetabolic metastatic lesions in the liver and axial skeleton.",
     "A whole-body PET scan showed active cancer spreading to the liver and spine bones.", "Oncology"),
    ("The patient has immune thrombocytopenia characterized by severe ecchymosis and mucosal bleeding.",
     "The patient has an autoimmune condition causing very low blood clotting cells, leading to large bruises and bleeding.", "Oncology"),
    ("Laboratory testing confirmed severe microcytic hypochromic anemia secondary to chronic occult blood loss.",
     "Blood tests confirmed severe anemia with tiny, pale red blood cells caused by hidden, ongoing bleeding.", "Hematology"),
    ("The patient was administered recombinant granulocyte colony-stimulating factor to stimulate leukopoiesis.",
     "The patient was given a special protein shot to help the bone marrow produce new infection-fighting white blood cells.", "Hematology"),
    ("The patient has hereditary spherocytosis accompanied by chronic hemolytic anemia and splenomegaly.",
     "The patient has an inherited disorder where round red blood cells break down too fast, causing anemia and an enlarged spleen.", "Hematology"),
    ("Serum protein electrophoresis and immunofixation revealed an IgG kappa monoclonal gammopathy.",
     "A protein blood test revealed an abnormal clone of protein produced by plasma cells, pointing to multiple myeloma.", "Oncology"),
    ("Palliative radiation therapy was initiated to alleviate intractable pain from osseous metastases.",
     "Targeted radiation treatment was started to relieve severe bone pain caused by cancer spreading to the skeleton.", "Oncology"),

    # Endocrinology & Metabolism
    ("The patient presented in diabetic ketoacidosis with severe hyperglycemia, ketonuria, and tachypnea.",
     "The patient arrived with a life-threatening diabetes complication causing very high blood sugar, toxic acids in urine, and rapid breathing.", "Endocrinology"),
    ("Thyroid ultrasound revealed a solid hypoechoic nodule with microcalcifications and irregular margins.",
     "An ultrasound of the thyroid gland showed a solid dark lump with tiny calcium spots and ragged edges.", "Endocrinology"),
    ("The patient has primary hyperparathyroidism resulting in hypercalcemia, nephrolithiasis, and osteopenia.",
     "Overactive parathyroid glands caused excessive calcium in the blood, leading to kidney stones and thinning bones.", "Endocrinology"),
    ("Laboratory evaluation demonstrated primary adrenal insufficiency with hyponatremia and hyperpigmentation.",
     "Hormone tests showed the adrenal glands were underperforming, causing low sodium levels and skin darkening.", "Endocrinology"),
    ("Glycated hemoglobin was measured at eleven point two percent, reflecting suboptimal long-term glycemic control.",
     "The three-month average blood sugar test (A1C) was over eleven percent, showing very poorly controlled diabetes.", "Endocrinology"),
    ("The patient developed acute symptomatic hypoglycemia manifested by diaphoresis, tremors, and neuroglycopenia.",
     "The patient had a sudden severe drop in blood sugar causing heavy sweating, shaking, and confusion.", "Endocrinology"),
    ("Radioactive iodine uptake scan demonstrated diffuse toxic goiter diagnostic of Graves' disease.",
     "A specialized thyroid scan showed the entire thyroid gland was overactive, confirming Graves' disease.", "Endocrinology"),
    ("The patient was diagnosed with severe primary hypothyroidism and started on levothyroxine replacement therapy.",
     "The patient was diagnosed with an underactive thyroid gland and prescribed daily thyroid hormone replacement tablets.", "Endocrinology"),
    ("Pituitary MRI identified a microadenoma secreting excessive adrenocorticotropic hormone, causing Cushing's syndrome.",
     "A pituitary scan found a tiny benign tumor making too much stress hormone, causing weight gain and high cortisol.", "Endocrinology"),
    ("Serum uric acid levels were significantly elevated, triggering acute podagra in the first metatarsophalangeal joint.",
     "High uric acid crystals in the blood caused a sudden, intensely painful gout attack in the big toe.", "Endocrinology"),

    # Infectious Diseases
    ("Blood cultures yielded Gram-negative bacteremia consistent with Escherichia coli urosepsis.",
     "Blood tests found bacteria growing in the bloodstream from a severe urinary tract infection that spread.", "Infectious Disease"),
    ("The patient was treated for severe Clostridioides difficile colitis with oral vancomycin.",
     "The patient was treated for a severe bacterial bowel infection caused by antibiotic use with oral medication.", "Infectious Disease"),
    ("Lumbar puncture confirmed acute bacterial meningitis with elevated neutrophils and low glucose.",
     "A spinal fluid test confirmed a serious bacterial infection around the brain and spinal cord.", "Infectious Disease"),
    ("Chest imaging demonstrated cavitary lesions in the right upper lobe suggestive of active pulmonary tuberculosis.",
     "Chest scans showed hollowed-out holes in the upper right lung typical of active tuberculosis infection.", "Infectious Disease"),
    ("The patient developed necrotizing fasciitis of the lower extremity requiring emergent surgical debridement.",
     "The patient had a rapid, flesh-eating bacterial infection in the leg that required immediate emergency surgery to cut away dying tissue.", "Infectious Disease"),
    ("Serological testing revealed acute hepatitis B infection with positive surface antigen and high viral load.",
     "Blood antibody tests showed a fresh hepatitis B virus infection actively multiplying in the liver.", "Infectious Disease"),
    ("The patient presented with septic shock characterized by profound refractory hypotension and lactic acidosis.",
     "The patient had a severe whole-body infection causing dangerously low blood pressure that did not respond to fluids.", "Infectious Disease"),
    ("Dermatological inspection identified painful vesicular eruptions in a dermatomal distribution, diagnostic of herpes zoster.",
     "A skin exam found a painful rash of fluid-filled blisters along a single nerve band, confirming shingles.", "Infectious Disease"),
    ("The patient was initiated on empirical broad-spectrum carbapenem therapy for multidrug-resistant nosocomial infection.",
     "The patient was started on a strong hospital-grade antibiotic to fight bacteria resistant to standard medicines.", "Infectious Disease"),
    ("Nasopharyngeal swab RT-PCR was positive for influenza A virus subtype H1N1.",
     "A nose swab genetic test confirmed the patient had the seasonal swine flu virus.", "Infectious Disease"),

    # Orthopedics & Rheumatology
    ("Radiographs revealed a comminuted intra-articular fracture of the distal radius requiring open reduction and internal fixation.",
     "X-rays showed the wrist bone was shattered into multiple pieces inside the joint, requiring surgery with plates and screws.", "Orthopedics"),
    ("The patient has severe bilateral knee osteoarthritis with bone-on-bone articulation and osteophyte formation.",
     "The patient has advanced joint wear-and-tear in both knees, with worn-out cartilage and extra bone spurs rubbing together.", "Orthopedics"),
    ("Serological testing demonstrated high-titer rheumatoid factor and anti-citrullinated protein antibodies.",
     "Blood tests showed high levels of immune proteins that attack the body's own joints, confirming rheumatoid arthritis.", "Rheumatology"),
    ("Dual-energy X-ray absorptiometry revealed severe osteoporosis with a T-score of minus three point two.",
     "A bone density scan showed weak, brittle bones with a high risk of breaking from minor falls.", "Orthopedics"),
    ("Pelvic imaging identified an undisplaced subcapital femoral neck fracture following a low-energy ground-level fall.",
     "A hip scan showed a crack right below the ball of the hip joint after falling from standing height.", "Orthopedics"),
    ("Knee MRI demonstrated a complex tear of the posterior horn of the medial meniscus.",
     "A knee MRI showed a ragged tear in the inner rubbery shock-absorbing cartilage of the knee.", "Orthopedics"),
    ("The patient suffers from ankylosing spondylitis characterized by sacroiliitis and spinal syndesmophytes.",
     "The patient has an inflammatory arthritis that fuses the spine bones and inflames the pelvic joints.", "Rheumatology"),
    ("Shoulder ultrasound revealed full-thickness supraspinatus tendon tear with subacromial bursitis.",
     "An ultrasound of the shoulder showed a complete tear in the main rotator cuff tendon along with an inflamed fluid cushion.", "Orthopedics"),
    ("Arthrocentesis of the knee joint yielded cloudy synovial fluid with monosodium urate crystals.",
     "Drawing fluid from the swollen knee revealed cloudy joint fluid filled with sharp gout crystals.", "Rheumatology"),
    ("The patient underwent total hip arthroplasty with an uncemented ceramic-on-polyethylene prosthesis.",
     "The patient had a complete hip replacement surgery using a modern ceramic ball and plastic socket implant.", "Orthopedics"),

    # Dermatology & Ophthalmology
    ("Skin biopsy of the suspicious pigmented lesion confirmed invasive nodular malignant melanoma.",
     "A small skin sample of the dark mole confirmed a serious, fast-spreading type of skin cancer.", "Dermatology"),
    ("Slit-lamp biomicroscopy demonstrated nuclear sclerotic cataract causing significant visual acuity deterioration.",
     "An eye microscope exam showed clouding and yellowing of the eye lens, causing blurry vision.", "Ophthalmology"),
    ("Fundoscopic examination revealed proliferative diabetic retinopathy with neovascularization and vitreous hemorrhage.",
     "Looking into the back of the eye showed new, fragile blood vessels leaking blood into the eye fluid due to diabetes.", "Ophthalmology"),
    ("Tonometry revealed elevated intraocular pressure of thirty-two millimeters of mercury, indicating acute angle-closure glaucoma.",
     "Measuring eye pressure showed a dangerous spike causing emergency glaucoma that threatens eyesight.", "Ophthalmology"),
    ("The patient has severe plaque psoriasis affecting forty percent of total body surface area.",
     "The patient has thick, red, scaly skin patches covering nearly half of their body.", "Dermatology"),
    ("The ophthalmologist diagnosed rhegmatogenous retinal detachment requiring immediate vitrectomy and gas tamponade.",
     "The eye surgeon found the light-sensitive retina peeled off the back of the eye, requiring urgent surgery to reattach it.", "Ophthalmology"),
    ("The patient suffers from atopic dermatitis with severe pruritus and secondary lichenification from chronic excoriation.",
     "The patient has severe eczema with intense itching and thickened skin caused by constant scratching.", "Dermatology"),
    ("Fluorescein angiography revealed choroidal neovascularization secondary to wet age-related macular degeneration.",
     "A dye scan of the eye showed abnormal leaking blood vessels damaging the central part of vision.", "Ophthalmology"),
    ("The lesion was diagnosed as basal cell carcinoma with infiltrative histology, indicated for Mohs micrographic surgery.",
     "The skin spot was identified as a common skin cancer that needs precise layer-by-layer surgery to remove all roots.", "Dermatology"),
    ("Optical coherence tomography demonstrated marked cystoid macular edema with retinal thickening.",
     "A high-tech laser scan of the eye showed swelling and fluid bubbles in the central vision zone.", "Ophthalmology")
]

# Clinical expansion templates to generate systematically varied, synthetically augmented clinical scenarios (~800 pairs)
CLINICAL_VARIANTS = [
    ("Initial admission evaluation: {med}", "Hospital admission note: {plain}"),
    ("The emergency department physician documented that {med_l}", "The ER doctor noted that {plain_l}"),
    ("Discharge summary notation: {med}", "Take-home medical summary: {plain}"),
    ("Clinical assessment reveals: {med}", "Medical checkup result: {plain}"),
    ("Follow-up outpatient note: {med}", "Clinic visit note: {plain}"),
    ("Diagnostic imaging and laboratory findings: {med}", "Test results explain: {plain}"),
    ("The consulting specialist confirmed: {med}", "The specialist doctor confirmed: {plain}"),
    ("Prior to initiating therapy, records indicate: {med}", "Before starting treatment, the records show: {plain}"),
]

def generate_medical_dataset():
    """Generates 800 high-quality medical simplification pairs."""
    dataset = []
    pair_id = 1

    # First add all original base entries
    for med, plain, cat in MEDICAL_ENTRIES:
        dataset.append({
            "id": f"med_{pair_id:04d}",
            "domain": "medical_simplification",
            "category": cat,
            "input_text": f"Simplify technical medical language into plain English: {med}",
            "target_text": plain
        })
        pair_id += 1

    # Augment systematically with realistic clinical prefixes and framing to reach ~800 pairs
    for med, plain, cat in MEDICAL_ENTRIES:
        med_clean = med.strip()
        plain_clean = plain.strip()
        med_l = med_clean[0].lower() + med_clean[1:]
        plain_l = plain_clean[0].lower() + plain_clean[1:]

        for t_med, t_plain in CLINICAL_VARIANTS:
            inp = t_med.format(med=med_clean, med_l=med_l)
            tgt = t_plain.format(plain=plain_clean, plain_l=plain_l)
            dataset.append({
                "id": f"med_{pair_id:04d}",
                "domain": "medical_simplification",
                "category": cat,
                "input_text": f"Simplify technical medical language into plain English: {inp}",
                "target_text": tgt
            })
            pair_id += 1
            if len(dataset) >= 800:
                break
        if len(dataset) >= 800:
            break

    random.shuffle(dataset)
    return dataset

# -----------------------------------------------------------------------------
# General-Language Benchmark (100 strictly held-out prompts)
# Spanning 3 categories:
# 1. General sentence simplification / summarization (35)
# 2. General factual & commonsense QA / reasoning (35)
# 3. Creative / everyday instruction following (30)
# -----------------------------------------------------------------------------
GENERAL_BENCHMARK = [
    # Category 1: General text simplification / rewriting
    ("Simplify this sentence: The meteorological conditions precipitated an unexpected cancellation of the aviation departures.",
     "Bad weather caused the flights to be canceled.", "general_simplification"),
    ("Simplify this sentence: Subsequent to the fiscal quarter conclusion, the enterprise implemented personnel downsizing.",
     "After the business quarter ended, the company laid off workers.", "general_simplification"),
    ("Simplify this sentence: The velocity of the vehicular collision caused extensive structural distortion to the chassis.",
     "The car crashed at high speed and suffered heavy frame damage.", "general_simplification"),
    ("Simplify this sentence: Prior to consumption, culinary ingredients must be subjected to thermal purification.",
     "Food must be cooked thoroughly before eating.", "general_simplification"),
    ("Simplify this sentence: The pedagogical institution inaugurated a novel curriculum focusing on computational sciences.",
     "The school launched a new computer science program.", "general_simplification"),
    ("Simplify this sentence: Atmospheric precipitation was negligible throughout the entirety of the summer season.",
     "Very little rain fell all summer.", "general_simplification"),
    ("Simplify this sentence: The residential dwelling underwent extensive architectural renovations prior to commercial listing.",
     "The house was remodeled before being put up for sale.", "general_simplification"),
    ("Simplify this sentence: The magistrate rendered an impartial determination grounded strictly in jurisprudential statutes.",
     "The judge made a fair decision based strictly on the law.", "general_simplification"),
    ("Simplify this sentence: Individuals displaying symptoms of profound sleep deprivation demonstrate diminished cognitive reflexes.",
     "People who do not get enough sleep react more slowly.", "general_simplification"),
    ("Simplify this sentence: The botanical garden preserves flora endemic to the sub-Saharan continental region.",
     "The botanical garden protects plants native to sub-Saharan Africa.", "general_simplification"),
    ("Simplify this sentence: The aquatic expedition encountered severe maritime turbulence during nocturnal navigation.",
     "The boat faced rough seas while sailing at night.", "general_simplification"),
    ("Simplify this sentence: Economic inflation diminished consumer purchasing power across multiple commercial sectors.",
     "Rising prices made it harder for people to buy everyday goods.", "general_simplification"),
    ("Simplify this sentence: The municipality instituted mandatory recycling protocols to curtail landfill accumulation.",
     "The city made recycling mandatory to reduce garbage in landfills.", "general_simplification"),
    ("Simplify this sentence: Prolonged solar radiation exposure without dermal sunscreen induces epidermal cellular damage.",
     "Staying in the sun too long without sunscreen burns and damages the skin.", "general_simplification"),
    ("Simplify this sentence: The conductor terminated the locomotive journey due to an insurmountable track obstruction.",
     "The train driver stopped the journey because an obstacle blocked the tracks.", "general_simplification"),
    ("Simplify this sentence: The electrical grid experienced a temporary disruption following lightning strikes.",
     "A lightning storm knocked out the power temporarily.", "general_simplification"),
    ("Simplify this sentence: Subterranean seismic activity registered a magnitude of six point four on the measurement scale.",
     "An underground earthquake measured 6.4 on the scale.", "general_simplification"),
    ("Simplify this sentence: The ornithological sanctuary accommodates migrating birds traversing hemispheric flyways.",
     "The bird sanctuary shelters birds flying south for the winter.", "general_simplification"),
    ("Simplify this sentence: The culinary artisan seasoned the broth with indigenous aromatic spices.",
     "The chef added local spices to give the soup good flavor.", "general_simplification"),
    ("Simplify this sentence: The automotive technician restored optimal braking function by exchanging depleted friction pads.",
     "The mechanic fixed the car brakes by replacing worn brake pads.", "general_simplification"),
    ("Simplify this sentence: Academic researchers investigated the correlation between dietary sugar intake and dental decay.",
     "Researchers studied how eating sugary foods causes cavities.", "general_simplification"),
    ("Simplify this sentence: The theatrical production incorporated synchronized visual projections and acoustic amplification.",
     "The play used matching video projections and sound speakers.", "general_simplification"),
    ("Simplify this sentence: Prolonged periods of drought diminished reservoir water reserves below historic minimums.",
     "A long dry spell left water reservoirs almost completely empty.", "general_simplification"),
    ("Simplify this sentence: The committee finalized their consensus subsequent to prolonged deliberative exchanges.",
     "The committee reached an agreement after a long discussion.", "general_simplification"),
    ("Simplify this sentence: The celestial observatory documented a novel comet traversing the solar perimeter.",
     "The space observatory spotted a new comet passing through the solar system.", "general_simplification"),
    ("Simplify this sentence: Urban planners prioritized pedestrian thoroughfares to decrease vehicular congestion.",
     "City planners built walking paths to reduce traffic jams.", "general_simplification"),
    ("Simplify this sentence: The maritime port authorities prohibited vessel departure due to gale-force gusts.",
     "Port officials stopped ships from leaving port during the strong wind storm.", "general_simplification"),
    ("Simplify this sentence: The software patch rectified several operational vulnerabilities within the operating system.",
     "The software update fixed multiple security bugs in the system.", "general_simplification"),
    ("Simplify this sentence: Modern agricultural combines maximize wheat harvesting efficiency across extensive acreage.",
     "Modern harvesting machines make gathering wheat fast and efficient on large farms.", "general_simplification"),
    ("Simplify this sentence: The historian deciphered ancient hieroglyphic inscriptions carved into limestone slabs.",
     "The historian translated old picture writings carved into stone.", "general_simplification"),
    ("Simplify this sentence: Cellular communication towers maintain signal coverage across rural territories.",
     "Cell towers keep mobile phone service running in rural areas.", "general_simplification"),
    ("Simplify this sentence: Synthetic fertilizers accelerate vegetation maturity but may contaminate hydrological basins.",
     "Chemical fertilizers help crops grow faster but can pollute rivers and lakes.", "general_simplification"),
    ("Simplify this sentence: The mountaineering expedition ascended the treacherous ridgeline during inclement blizzard conditions.",
     "The climbers walked up the dangerous mountain ridge during a severe snowstorm.", "general_simplification"),
    ("Simplify this sentence: The art conservationist restored the seventeenth-century oil painting to its pristine state.",
     "The art restorer carefully cleaned the 17th-century painting to look new.", "general_simplification"),
    ("Simplify this sentence: International postal consignments require customs documentation prior to cross-border dispatch.",
     "International packages need customs forms before they can be shipped abroad.", "general_simplification"),

    # Category 2: General factual & commonsense QA
    ("Answer this question: Why do leaves change color in the autumn?",
     "Leaves change color in autumn because trees stop producing green chlorophyll as sunlight decreases, revealing yellow and red pigments.", "commonsense_qa"),
    ("Answer this question: What is the primary purpose of bees in agriculture?",
     "Bees pollinate flowers and crops, which helps fruits, vegetables, and seeds grow.", "commonsense_qa"),
    ("Answer this question: Why is the ocean salty?",
     "The ocean is salty because rivers wash minerals and dissolved salts from rocks on land into the sea.", "commonsense_qa"),
    ("Answer this question: How does a thermos keep hot liquids warm?",
     "A thermos uses a double-walled vacuum chamber to stop heat from escaping through conduction and convection.", "commonsense_qa"),
    ("Answer this question: What causes thunder after a flash of lightning?",
     "Lightning rapidly superheats the surrounding air, making it expand explosively and create a shockwave of sound.", "commonsense_qa"),
    ("Answer this question: Why do we have leap years every four years?",
     "Leap years add an extra day because Earth takes about 365.25 days to orbit the sun, keeping our calendar aligned.", "commonsense_qa"),
    ("Answer this question: What makes bread dough rise during baking?",
     "Yeast feeds on sugars in the flour and releases carbon dioxide gas bubbles that get trapped and expand the dough.", "commonsense_qa"),
    ("Answer this question: Why do ice cubes float on liquid water?",
     "Ice floats because water expands when it freezes, making solid ice less dense than liquid water.", "commonsense_qa"),
    ("Answer this question: How do airplanes generate lift to stay airborne?",
     "Airplane wings are curved on top, making air move faster over the top and creating higher pressure underneath that pushes the plane up.", "commonsense_qa"),
    ("Answer this question: Why do camels have humps on their backs?",
     "Camels store fat in their humps, which their bodies can convert into energy and water when food is scarce in the desert.", "commonsense_qa"),
    ("Answer this question: What causes the ocean tides on Earth?",
     "Ocean tides are caused primarily by the gravitational pull of the moon and sun on Earth's oceans.", "commonsense_qa"),
    ("Answer this question: Why do humans yawn when they are tired?",
     "Yawning helps cool the brain and increases alertness by bringing in a deep breath of fresh oxygen.", "commonsense_qa"),
    ("Answer this question: How does a refrigerator keep food cold?",
     "A refrigerator pumps chemical refrigerant through coils to absorb heat from inside the compartment and release it outside.", "commonsense_qa"),
    ("Answer this question: Why is the sky blue during a clear day?",
     "The sky is blue because Earth's atmosphere scatters short blue wavelengths of sunlight more than other colors.", "commonsense_qa"),
    ("Answer this question: What is the function of the ozone layer in Earth's atmosphere?",
     "The ozone layer absorbs and blocks harmful ultraviolet radiation from the sun, protecting life on Earth.", "commonsense_qa"),
    ("Answer this question: How do mirrors reflect our image?",
     "Mirrors have a smooth silver or aluminum backing that reflects incoming light rays back at the exact same angle.", "commonsense_qa"),
    ("Answer this question: Why does metal feel colder than wood at the same room temperature?",
     "Metal conducts heat away from your skin much faster than wood, making it feel colder to the touch.", "commonsense_qa"),
    ("Answer this question: What is the purpose of windshield wipers on an automobile?",
     "Windshield wipers clear rain, snow, and dirt from the windshield so the driver can see the road clearly.", "commonsense_qa"),
    ("Answer this question: Why do birds migrate south during the winter?",
     "Birds migrate south to find warmer weather, open water, and abundant food sources when winter freezes northern areas.", "commonsense_qa"),
    ("Answer this question: How does a compass needle find north?",
     "A compass needle is a small magnet that aligns itself with Earth's natural magnetic field pointing toward the magnetic poles.", "commonsense_qa"),
    ("Answer this question: What is the main gas found in Earth's atmosphere?",
     "Nitrogen makes up approximately seventy-eight percent of Earth's atmosphere.", "commonsense_qa"),
    ("Answer this question: Why do onions make your eyes water when you chop them?",
     "Cutting an onion releases sulfur compounds that mix with natural tear moisture to create a mild acid that irritates the eyes.", "commonsense_qa"),
    ("Answer this question: How do solar panels generate electricity from sunlight?",
     "Solar panels use silicon photovoltaic cells that knock electrons free when struck by sunlight, creating an electric current.", "commonsense_qa"),
    ("Answer this question: Why do ships made of heavy steel float on water?",
     "Steel ships have hollow hulls filled with air, making the total density of the ship lighter than the water it displaces.", "commonsense_qa"),
    ("Answer this question: What makes fireflies produce light at night?",
     "Fireflies create light through a natural chemical reaction in their bellies called bioluminescence, using an enzyme named luciferase.", "commonsense_qa"),
    ("Answer this question: Why does boiling water bubble?",
     "Boiling water bubbles because the liquid turns into water vapor gas at the bottom of the pan and rises to the surface.", "commonsense_qa"),
    ("Answer this question: How do roots help a tree survive?",
     "Roots anchor the tree firmly in the ground and absorb water and essential soil nutrients.", "commonsense_qa"),
    ("Answer this question: What is the difference between a fruit and a vegetable botanically?",
     "A fruit develops from the flower of a plant and contains seeds, whereas vegetables are roots, stems, or leaves.", "commonsense_qa"),
    ("Answer this question: Why does a rainbow appear after rain?",
     "Raindrops act like tiny glass prisms that bend and reflect sunlight, separating white light into its distinct rainbow colors.", "commonsense_qa"),
    ("Answer this question: How does soap clean grease off hands?",
     "Soap molecules have one end that binds to water and another that binds to grease, trapping dirt so water can wash it away.", "commonsense_qa"),
    ("Answer this question: Why do cats purr?",
     "Cats purr to communicate contentment, self-soothe when stressed, or bond with their mothers.", "commonsense_qa"),
    ("Answer this question: What causes earthquakes to happen?",
     "Earthquakes occur when stress builds up along tectonic fault lines in Earth's crust until rocks break and slip suddenly.", "commonsense_qa"),
    ("Answer this question: Why is sleeping essential for human health?",
     "Sleep allows the brain to organize memories, clears metabolic waste, and lets the body repair muscles and tissues.", "commonsense_qa"),
    ("Answer this question: What is the function of roots on a mangrove tree?",
     "Mangrove roots stabilize coastline mud and breathe air through specialized pores during high tides.", "commonsense_qa"),
    ("Answer this question: Why does sound travel faster through water than through air?",
     "Water molecules are packed much closer together than air molecules, allowing sound pressure waves to pass from particle to particle faster.", "commonsense_qa"),

    # Category 3: Creative, instruction following, and everyday tasks
    ("Draft a polite thank-you note to a colleague who helped you finish a project on time.",
     "Thank you so much for your assistance with our project. Your help was invaluable in getting everything completed on time.", "instruction_following"),
    ("Write three simple rules for internet security that anyone can follow.",
     "1. Use strong, unique passwords for every account. 2. Enable two-factor authentication whenever available. 3. Never click suspicious links in unexpected emails.", "instruction_following"),
    ("Explain how to brew a cup of hot green tea in three steps.",
     "First, heat water until just before boiling. Second, place your green tea bag or leaves in a cup and pour the hot water over it. Third, let it steep for two to three minutes before removing.", "instruction_following"),
    ("Rewrite this sentence to sound friendly and warm: 'The meeting starts at two. Be there.'",
     "Looking forward to seeing you at our meeting today at 2:00 PM!", "instruction_following"),
    ("Give two practical tips for saving money on household electricity bills.",
     "Turn off lights and electronics when leaving a room, and unplug chargers when not in use.", "instruction_following"),
    ("Summarize the moral lesson of the fable 'The Tortoise and the Hare'.",
     "Consistent effort and patience win in the long run over overconfidence and laziness.", "instruction_following"),
    ("Write a brief out-of-office email notice for a two-day vacation.",
     "Thank you for reaching out. I am currently out of the office with limited email access and will return on Monday. For urgent matters, please contact support.", "instruction_following"),
    ("Provide three steps to plant a sunflower in a garden pot.",
     "1. Fill a pot with loose potting soil. 2. Plant the sunflower seed one inch deep and cover it with dirt. 3. Water gently and place the pot in direct sunlight.", "instruction_following"),
    ("Describe the appearance of a cozy autumn morning in two sentences.",
     "Golden sunlight filters through amber trees as a gentle crisp breeze stirs fallen leaves on the ground. A warm cup of cider steams quietly on the wooden porch table.", "instruction_following"),
    ("Convert this informal text message into a professional work message: 'hey cant make it 2day sick as a dog'",
     "Good morning. I am feeling unwell today and will need to take a sick day. I will keep you updated on my return.", "instruction_following"),
    ("List three ingredients needed to make homemade chocolate chip cookies.",
     "All-purpose flour, butter, and semi-sweet chocolate chips.", "instruction_following"),
    ("Write an encouraging encouraging note for a student taking their driving test today.",
     "Stay calm, take your time, and trust in your practice. You have got this!", "instruction_following"),
    ("Explain the difference between a book author and an illustrator in simple terms.",
     "The author writes the story using words, while the illustrator draws the pictures that bring the story to life visually.", "instruction_following"),
    ("Draft a polite one-sentence request asking a neighbor to turn down loud music late at night.",
     "Hi neighbor, would you mind turning down the music a bit so we can get some sleep tonight? Thank you so much.", "instruction_following"),
    ("List three benefits of drinking plenty of water every day.",
     "Drinking water keeps you hydrated, improves energy and focus, and supports healthy digestion.", "instruction_following"),
    ("Explain how to iron a dress shirt in three basic steps.",
     "Start by ironing the collar and cuffs flat, then iron the sleeves, and finish by pressing the front and back body panels smoothly.", "instruction_following"),
    ("Write a slogan for a neighborhood recycling campaign.",
     "Small daily choices create a cleaner tomorrow: Recycle today!", "instruction_following"),
    ("Explain the difference between fiction and non-fiction books.",
     "Fiction stories are created from the author's imagination, while non-fiction books describe real people, historical events, and true facts.", "instruction_following"),
    ("Give two helpful ideas for a fun family game night.",
     "Choose an easy-to-learn board game everyone can play, and make homemade popcorn together as a snack.", "instruction_following"),
    ("Write a short welcoming greeting for visitors entering a public library.",
     "Welcome to our library! Feel free to explore our book collections, study spaces, and community resources.", "instruction_following"),
    ("Provide three tips for keeping indoor house plants healthy.",
     "Provide adequate indirect sunlight, avoid overwatering, and ensure pots have drainage holes at the bottom.", "instruction_following"),
    ("Draft a short congratulatory message for a friend who completed their university degree.",
     "Huge congratulations on graduating! Your hard work and dedication have paid off, and I wish you all the best in your next chapter.", "instruction_following"),
    ("Describe the smell and sound of rain falling on a hot summer afternoon.",
     "The sharp scent of damp earth fills the warm air as heavy raindrops hiss rhythmically against the sun-baked pavement.", "instruction_following"),
    ("List three common items you should always pack when going hiking in the woods.",
     "A full water bottle, a map or compass, and a small first-aid kit.", "instruction_following"),
    ("Convert this passive sentence to active voice: 'The award was presented to Maria by the principal.'",
     "The principal presented the award to Maria.", "instruction_following"),
    ("Explain what a library card allows you to do.",
     "A library card lets you borrow books, audiobooks, and movies for free and access digital research databases.", "instruction_following"),
    ("Write two sentences describing a golden retriever puppy playing with a ball.",
     "The fluffy golden puppy bounded across the green lawn with boundless energy. He pounced on the bright red tennis ball with playful barks.", "instruction_following"),
    ("Draft a quick RSVP note accepting an invitation to a birthday party.",
     "Thank you for the invitation! I would love to come and look forward to celebrating your birthday on Saturday.", "instruction_following"),
    ("Give two simple suggestions to reduce food waste at home.",
     "Plan meals ahead of time with a grocery list, and save dinner leftovers in the fridge for lunch the next day.", "instruction_following"),
    ("Explain what a lighthouse does for ships at sea.",
     "A lighthouse beams a bright warning light to guide passing ships safely away from dangerous rocky coastlines.", "instruction_following")
]

def generate_general_benchmark():
    """Formats the 100 strictly held-out general prompts."""
    benchmark = []
    for idx, (prompt, target, cat) in enumerate(GENERAL_BENCHMARK, start=1):
        benchmark.append({
            "id": f"gen_{idx:03d}",
            "domain": "general_language",
            "category": cat,
            "input_text": prompt,
            "target_text": target
        })
    return benchmark

def main():
    print("Generating Medical Text Simplification Dataset...")
    medical_data = generate_medical_dataset()
    print(f"Total Medical Examples generated: {len(medical_data)}")

    # Split: Train (~80%), Val (~10%), Test (~10%)
    n_total = len(medical_data)
    n_test = 75
    n_val = 75
    n_train = n_total - n_test - n_val

    train_data = medical_data[:n_train]
    val_data = medical_data[n_train:n_train + n_val]
    test_data = medical_data[n_train + n_val:]

    print(f"Splits -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    def save_jsonl(data, filename):
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} items to {filepath}")

    save_jsonl(train_data, "medical_train.jsonl")
    save_jsonl(val_data, "medical_val.jsonl")
    save_jsonl(test_data, "medical_test.jsonl")

    print("\nGenerating General-Language Benchmark (Held-out)...")
    general_benchmark = generate_general_benchmark()
    save_jsonl(general_benchmark, "general_eval_benchmark.jsonl")
    print(f"Total General-Language Benchmark items: {len(general_benchmark)}")

if __name__ == "__main__":
    main()

