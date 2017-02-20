
from __future__ import print_function
import os
import sys
import gzip
import ssl
import urllib
import urllib2
from six.moves.urllib.request import urlretrieve

home_dir = 'https://nrhm-mis.nic.in/hmisreports/AHSReports.aspx'

data_struct_ahs = 'https://nrhm-mis.nic.in/HMISReports/frmDownload.aspx?download=NzV0zz1EIlIPSnLil6jCINq1QSoYCR398XDhbVpx56boHkeoxMj+okY4lX2czEhP'
data_struct_cab = 'https://nrhm-mis.nic.in/HMISReports/frmDownload.aspx?download=NzV0zz1EIlIPSnLil6jCINq1QSoYCR398XDhbVpx56Z/AmpDILZu4UY4lX2czEhP'

comb_22_csv = 'https://nrhm-mis.nic.in/HMISReports/frmDownload.aspx?download=NzV0zz1EIlIPSnLil6jCINtpBpLXVF5OfwSbZvffsLzR8SjmRY8VKA=='

last_percent_reported = None


context = ssl._create_unverified_context()
# urllib.urlopen("https://no-valid-cert")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

file = urllib2.urlopen(data_struct_ahs, context=ctx)

with open('AHS_struct.xlsx','wb') as output:
  output.write(file.read())

# testfile = urllib.URLopener()
# testfile.retrieve(data_struct_ahs, 'AHS_struct.xlsx', context=ctx)























def download_progress_hook(count, blockSize, totalSize):
	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
		    sys.stdout.write("%s%%" % percent)
		    sys.stdout.flush()
		else:
		    sys.stdout.write(".")
		    sys.stdout.flush()
	  
		last_percent_reported = percent
        
def maybe_download(filename, force=False):
	if force or not os.path.exists(filename):
		print('Attempting to download:', filename) 
		filename, _ = urlretrieve(data_struct_ahs, filename, reporthook=download_progress_hook, context=context)
		print('\nDownload Complete!')
	statinfo = os.stat(filename)
	return filename

# maybe_download('AHS_struct.xlsx')