
```bash
uv run streamlit run src/music_hrv/gui/app.py  # Launch GUI
uv run pytest                                   # Run tests
uv run ruff check src/ tests/ --fix            # Lint
```


hier schreibe ich rein was zu tun ist für mich (nicht für CLAUDE)




- [x] mit den anderen ablären welche metriken sie brauchen (@2025-12-03 15:50)
- [x] checken wo die daten sind von denen der Marius geredet hat (@2026-01-14) ✅ 2026-01-11
- [x] VNS analyse reinbekommen ✅ 2025-12-22
- [x] group level analysis machen ✅ 2025-12-22
- [x] das tool den anderen vorstellen (@2025-12-02 17:00) und auch sagen was sie beachten sollen wenn sie das tool nicht verwenden -> duplicated events und duplicated RR-intervalls und measurement gaps und 
- [ ] für musical events es auch machen das es geht wenn nur eine grenze da ist
- [ ] generation von events die fehlen automatisch falls möglich (muss markiert werden)
- [x] es kann sein das die musik gestoppt wurde und das dann die messung über 90 minuten dauerte. es soll möglich sein diese Messungs/musikpausen zu vermerken mit events und das ohne die pausen die messung 90 minuten dauern sollte ✅ 2025-12-22
- [ ] Orthostasic reactions nennt man es wenn jemand die position ändert oder herumgeht
- [ ] das event labeling/automatic event dedection funktioniert nicht so gut für demo data
- [ ] r-r power spectrum (a nice plot i think)
- [ ] fix how artifactg rate is calculated! (use neurokit2 for this also?)
- [x] make the plots in the analysis section better ✅ 2025-12-22
	- [ ] the mean in the hr-distribution plot should look better
- [x] fix it so that the first_measurement section is properly analyzed ✅ 2026-01-11



### organisation

- check with Kubius if the pipeline is valid -> same results as with this pipeline
- send the project to josef tatschl so he can review it




Potential artifact causes and influencing variables depending on the principle of measurement Hardware and physiological considerations- Ectopic beats, atrial fibrillation- Electrophysiological signal strength - Measurement site of the application- Sensor or electrode geometry and contact (pressure)- Wavelength of light- Transmission of the light reflection Software considerations- Sampling rate- R-wave detection algorithm- Preprocessing of data (e.g., detrending method)- Artefact correction incl. digital filtering (e.g., electrophysio logical and motion artifacts)- Algorithms and default settings of signal processing (e.g., frequency bands) Considerations of measurement protocol incl. personal and en vironmental factors- Resting state or stress / exercise conditions (e.g., intensity, duration, modality)- Time of day (e.g., morning vs. night) and seasonal effects- Body position (e.g., lying vs. standing)- Duration of measurement and chosen HRV metric (time, frequency or non-linear domain)- Environment (e.g., temperature, humidity, sun and light exposure, hypoxia)- Age, sex and genetics- Hormone status (e.g., menstrual cycle, pregnancy)- Caffeine and alcohol intake- Medications- Bladder filling and hydration status- Nutrition intake- Health status (e.g., chronic disease, acute viral infection, food poisoning)- Acute and chronic stressors (e.g., occupation, travel, time shift and jetlag)- Physical activity and performance level- Body composition- Exercise load (e.g., intensity, duration) and recovery status- Injuries (e.g., impaired mobility) and pain status- Sleep quality and duration- Breathing pattern (e.g., paced breathing vs. natural rhythm and free running system)

https://www.germanjournalsportsmedicine.com/fileadmin/content/archiv2024/Issue_3/DtschZSportmed_10.5960dzsm.2024.595_Review_Gronwald_Heart_Rate_Variability_in_Sports_Medicine_and_Exercise_Science_2024-3.pdf