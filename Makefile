pyterrier:
	python src/codes/pyterrier_example.py

clean:
	rm -rf *_index/
	rm -rf */*_index/
	rm -rf index*/
	rm -rf .DS_Store
	rm -rf */.DS_Store
