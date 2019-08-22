#Parameters for Cleaning
Streaming_Folder="../HasilStreamingsss"
CleanOutFold=""
CleanOutFile="Vector"
echo "Creating Clean Tweets..."
#python3 Preprocessing-berita.py "$Streaming_Folder" $CleanOutFold "$CleanOutFile.pkl"

#Parameters for Processing
#DimRed="GRP" #Dimension Reduction GRP, SRP or LDA
metric="pmi" #evaluation metric: pmi, npmi or lcp
ref_corpus_dir="corpusWikiIdEdit/"
wordcount_file="results/wordcount/wc-oc.txt"

echo "Detecting topics & Calculating Coherence"
for DimRed in "GRP" "SVD"
do
	OutFoldTop="HasilTopicDetectionBerita-${DimRed^^}-Kernel"
	for i1 in 2 4 6 8 10 12 14 16 18 20
	do
		timing="${DimRed^^}-$i1"
		echo "${DimRed^^}-$i1"
		for i2 in 1 2 3 4 5
		do
			cd ../BeritaScriptDataBaru
			OutFileTop="beritaNoStem-$i1-$i2.txt"
			python3 Processinge.py $CleanOutFile $i1 "1" $DimRed $OutFoldTop $OutFileTop $timing
			cd ../Coherence
			topic_file="../BeritaScriptDataBaru/$OutFoldTop/$OutFileTop"
			oc_file="../BeritaScriptDataBaru/results/berita-${DimRed^^}-Kernel/$OutFileTop"
			python3 ComputeWordCount.py $topic_file $ref_corpus_dir > $wordcount_file
			python3 ComputeObservedCoherence.py $topic_file $metric $wordcount_file > $oc_file
			if [[ "$DimRed" =~ ^(LDA|NMF)$ ]]
			then
				break
			fi
		done
	done
done 