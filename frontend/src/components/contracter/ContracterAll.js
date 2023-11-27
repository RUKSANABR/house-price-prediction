import {View, Text, StyleSheet, TouchableOpacity} from 'react-native';
import React, {useState, useEffect} from 'react';
import firestore from '@react-native-firebase/firestore';
import {Avatar, Button, Card} from '@rneui/base';
import {useNavigation} from '@react-navigation/native';

export default function PropertyAll(props) {
  const navigation = useNavigation();
  const [data, setData] = useState([]);
  const userD = props.userD;
  useEffect(() => {
    const fetchData = async () => {
      try {
        const propertyRef = firestore()
          .collection('property')
          .where('status', '==', 'Available');
        const querySnapshot = await propertyRef.get();
        const dataArray = [];
        let docIndex = 0;

        querySnapshot.forEach(doc => {
          // doc.data() is never undefined for query doc snapshots
          dataArray.push({propertyId: doc.id, ...doc.data()});
          docIndex++;
        });

        setData(dataArray);
      } catch (error) {
        console.error('Error getting documents: ', error);
      }
    };

    fetchData(); // Call the asynchronous function here
  }, []);

  return (
    <View>
      <View style={{marginBottom: 70}}>
        {data &&
          data.map((propertyData, index) => {
            // Calculate the number of days ago
            const postedDate = new Date(propertyData.Date);
            const currentDate = new Date();
            const daysAgo = Math.floor(
              (currentDate - postedDate) / (1000 * 60 * 60 * 24),
            );
            //
            const timestamp = new Date(propertyData.Date);
            const year = timestamp.getFullYear();
            const month = (timestamp.getMonth() + 1)
              .toString()
              .padStart(2, '0'); // Months are zero-based
            const day = timestamp.getDate().toString().padStart(2, '0');
            const formattedDate = `${day}-${month}-${year}`;
            //

            return (
              <Card key={index} containerStyle={styles.cards}>
                <View style={styles.smallCard}>
                  <TouchableOpacity
                    onPress={() =>
                      navigation.navigate('Property', {userD, propertyData})
                    }>
                    <View style={styles.cardContent}>
                      <Avatar
                        source={require('../../assets/user.png')}
                        size={30}
                      />
                      <Text style={styles.cardText}>
                        {propertyData.userName}
                      </Text>
                    </View>
                    <View style={styles.cardContent}>
                      <TouchableOpacity style={styles.btn1}>
                        <Text style={styles.btnText}>{propertyData.LotArea} sq.ft</Text>
                      </TouchableOpacity>
                      <TouchableOpacity style={styles.btn1}>
                        <Text style={styles.btnText}>{ formattedDate}</Text>
                      </TouchableOpacity>
                    </View>
                    <View style={styles.textContent}>
                      <Text style={{color: 'black'}}>{propertyData.TotRmsAbvGrd} Rooms</Text>
                      <View style={{flexDirection:'row',alignItems:'center'}}>
                        <Avatar source={require('../../assets/phone.png')} size={30}/>
                      <Text style={{color: 'black'}}>
                         {propertyData.phone}
                      </Text>
                      
                      </View>
                      <Text style={{textAlign:'justify',color:'black',marginTop:10}}>User wants the help to build house.</Text>
                    </View>
                  </TouchableOpacity>
                </View>
                <View
                  style={[
                    styles.cardBottomContent,
                    {justifyContent: 'space-between'},
                  ]}>
                  <Text style={styles.cardText}>Posted {daysAgo} days ago</Text>
                  <Text style={styles.cardText}>₹{propertyData.price}</Text>
                </View>
              </Card>
            );
          })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  cards: {
    width: '95%',
    height: 240,
    backgroundColor: 'black',
    borderTopRightRadius: 21,
    borderTopLeftRadius: 21,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
    marginBottom: 20,
    alignSelf: 'center',
  },

  smallCard: {
    width: '108.5%',
    height: 200,
    backgroundColor: '#aedbeb',
    alignSelf: 'center',
    marginTop: -15,
    borderTopRightRadius: 20,
    borderTopLeftRadius: 20,
    padding: 10,
  },
  cardContent: {
    justifyContent: 'flex-start',
    flexDirection: 'row',
  },
  cardText: {
    fontSize: 18,
    fontWeight: '500',
    color: 'white',
    padding: 5,
    marginLeft: 10,
    marginRight: 5,
  },
  btn1: {
    backgroundColor: '#f6f0c1',
    width: 80,
    height: 39,
    borderRadius: 20,
    alignItems: 'center', // Center horizontally
    justifyContent: 'center',
    marginRight: 10,
  },
  btnText: {
    fontWeight: '500',
    color: 'white',
  },
  textContent: {
    justifyContent: 'center',
    marginLeft: 10,
    marginRight: 10,
    marginTop: 10,
  },
  cardBottomContent: {
    marginTop: 20,
    backgroundColor: 'black',
    width: '108%',
    height: 50,
    borderBottomLeftRadius: 25,
    borderBottomRightRadius: 25,
    alignSelf: 'center',
    flexDirection: 'row',
  },
});
