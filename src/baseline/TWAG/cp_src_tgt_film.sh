DOMAIN="film"
TEXT="/data1/tsq/TWAG/raw_dataset/"$DOMAIN
extract_path="/data1/tsq/contrastive/clust_documents/"$DOMAIN"/bart/test_as_valid/inverse_add0/ws_0.75/inverse_add_title/"
cp $extract_path'test.source' $TEXT'/test.src'
cp $extract_path'test.source' $TEXT'/valid.src'
cp $extract_path'train.source' $TEXT'/train.src'

cp $extract_path'test.target' $TEXT'/test.tgt'
cp $extract_path'test.target' $TEXT'/valid.tgt'
cp $extract_path'train.target' $TEXT'/train.tgt'


echo 999999 > $TEXT'/ignoredIndices.log'
echo 999999 > $TEXT'/test_ignoredIndices.log'
echo 999999 > $TEXT'/valid_ignoredIndices.log'